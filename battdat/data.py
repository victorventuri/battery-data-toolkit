"""Objects that represent battery datasets"""
import shutil
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Union, Optional, Collection, List, Dict, Set, Iterator, Tuple

from pandas import HDFStore
from pydantic import BaseModel, ValidationError
from pyarrow import parquet as pq
from pyarrow import Table
from tables import Group
import pandas as pd
import h5py

from battdat.schemas import BatteryMetadata
from battdat.schemas.column import RawData, CycleLevelData, ColumnSchema
from battdat.schemas.eis import EISData
from battdat import __version__

_default_schemas = {
    'raw_data': RawData(),
    'cycle_stats': CycleLevelData(),
    'eis_data': EISData()
}
"""Mapping between pre-defined datasets and schema"""

logger = logging.getLogger(__name__)


class BatteryDataset:
    """Holder for all data associated with tests for a battery.

    Attributes of this class define different view of the data (e.g., raw time-series, per-cycle statistics)
    or different types of data (e.g., EIS) along with the metadata for the class"""

    raw_data: Optional[pd.DataFrame] = None
    """Time-series data capturing the state of the battery as a function of time"""

    cycle_stats: Optional[pd.DataFrame] = None
    """Summary statistics of each cycle"""

    eis_data: Optional[pd.DataFrame] = None
    """Electrochemical Impedance Spectroscopy (EIS) data"""

    metadata: BatteryMetadata
    """Metadata for the battery construction and testing"""
    schemas: Dict[str, ColumnSchema]
    """Schemas for the data in each of the constituent data frames"""

    def __init__(self, metadata: Union[BatteryMetadata, dict] = None,
                 raw_data: Optional[pd.DataFrame] = None,
                 cycle_stats: Optional[pd.DataFrame] = None,
                 eis_data: Optional[pd.DataFrame] = None,
                 schemas: Optional[Dict[str, ColumnSchema]] = None):
        """

        Args:
            metadata: Metadata that describe the battery construction, data provenance and testing routines
            raw_data: Time-series data of the battery state
            cycle_stats: Summaries of each cycle
            eis_data: EIS data taken at multiple times
            schemas: Schemas describing each of the tabular datasets
        """
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, BaseModel):
            metadata = metadata.model_dump()

        # Provide schemas for each of the columns
        self.schemas = _default_schemas if schemas is None else schemas

        # Warn if the version of the metadata is different
        version_mismatch = False
        if (supplied_version := metadata.get('version', __version__)) != __version__:
            version_mismatch = True
            warnings.warn(f'Metadata was created in a different version of battdat. supplied={supplied_version}, current={__version__}.')

        try:
            self.metadata = BatteryMetadata(**metadata)
        except ValidationError:
            if version_mismatch:
                warnings.warn('Metadata failed to validate, probably due to version mismatch. Discarding until we support backwards compatibility')
                self.metadata = BatteryMetadata()
            else:
                raise
        self.raw_data = raw_data
        self.cycle_stats = cycle_stats
        self.eis_data = eis_data

    def validate_columns(self, allow_extra_columns: bool = True):
        """Determine whether the column types are appropriate

        Args:
            allow_extra_columns: Whether to allow unexpected columns

        Raises
            (ValueError): If the dataset fails validation
        """
        for attr_name, schema in self.schemas.items():
            data = getattr(self, attr_name)
            if data is not None:
                schema.validate_dataframe(data, allow_extra_columns)

    def validate(self) -> List[str]:
        """Validate the data stored in this object

        Ensures that the data are valid according to schemas and
        makes recommendations of improvements that one could make
        to increase the re-usability of the data.

        Returns:
            Recommendations to improve data re-use
        """
        self.validate_columns()
        output = []

        for attr_name, schema in self.schemas.items():
            data = getattr(self, attr_name)
            if data is not None:
                undefined = set(data.columns).difference(schema.column_names)
                output.extend([f'Undefined column, {u}, in {attr_name}. Add a description into schemas.{attr_name}.extra_columns'
                               for u in undefined])

        return output

    def to_hdf(self,
               path_or_buf: Union[str, Path, HDFStore],
               prefix: Optional[str] = None,
               append: bool = False,
               complevel: int = 0,
               complib: str = 'zlib'):
        """Save the data in the standardized HDF5 file format

        This function wraps the ``to_hdf`` function of Pandas and supplies fixed values for some options
        so that the data is written in a reproducible format.

        Args:
            path_or_buf: File path or HDFStore object.
            prefix: Prefix to use to differentiate this battery from (optionally) others stored in this HDF5 file
            append: Whether to clear any existing data in the HDF5 file before writing
            complevel: Specifies a compression level for data. A value of 0 disables compression.
            complib: Specifies the compression library to be used.
        """

        # Delete the old file if present
        if isinstance(path_or_buf, (str, Path)) and (Path(path_or_buf).is_file() and not append):
            Path(path_or_buf).unlink()

        # Create logic for adding metadata
        def add_metadata(f: Group, m: BaseModel):
            """Put the metadata in a standard location at the root of the HDF file"""
            metadata = m.model_dump_json()
            if append and 'metadata' in f._v_attrs:
                existing_metadata = f._v_attrs.metadata
                if metadata != existing_metadata:
                    warnings.warn('Metadata already in HDF5 differs from new metadata')
            f._v_attrs.metadata = metadata
            f._v_attrs.json_schema = m.model_json_schema()

        # Open the store
        if is_store := isinstance(path_or_buf, HDFStore):
            store = path_or_buf
        else:
            store = HDFStore(path_or_buf, complevel=complevel, complib=complib)

        try:
            # Store the various datasets
            #  Note that we use the "table" format to allow for partial reads / querying
            for key, schema in self.schemas.items():
                data = getattr(self, key)
                if data is not None:
                    if prefix is not None:
                        key = f'{prefix}_{key}'
                    data.to_hdf(path_or_buf, key=key, complevel=complevel,
                                complib=complib, append=False, format='table',
                                index=False)

                    # Write the schema
                    add_metadata(store.root[key], schema)

            # Store the high-level metadata
            add_metadata(store.root, self.metadata)
        finally:
            if not is_store:
                store.close()  # Close the store if we opened it

    @classmethod
    def from_hdf(cls,
                 path_or_buf: Union[str, Path, HDFStore],
                 subsets: Optional[Collection[str]] = None,
                 prefix: Union[str, None, int] = None) -> 'BatteryDataset':
        """Read the battery data from an HDF file

        Use :meth:`all_cells_from_hdf` to read all datasets from a file.

        Args:
            path_or_buf: File path or HDFStore object
            subsets : Which subsets of data to read from the data file (e.g., raw_data, cycle_stats)
            prefix: (``str``) Prefix designating which battery extract from this file,
                or (``int``) index within the list of available prefixes, sorted alphabetically.
                The default is to read the default prefix (``None``).

        """

        # Determine which datasets to read
        read_all = False
        if subsets is None:
            subsets = _default_schemas
            read_all = True

        # Open the store
        if is_store := isinstance(path_or_buf, HDFStore):
            store = path_or_buf
        else:
            store = HDFStore(path_or_buf, mode='r')

        try:
            # Determine which prefix to read, if an int is provided
            if isinstance(prefix, int):
                _, prefixes = cls.inspect_hdf(path_or_buf)
                prefix = sorted(prefixes)[prefix]

            data = {}
            schemas = {}
            for subset in subsets:
                # Prepend the prefix
                if prefix is not None:
                    key = f'{prefix}_{subset}'
                else:
                    key = subset

                try:
                    data[subset] = pd.read_hdf(path_or_buf, key)
                except KeyError as exc:
                    if read_all:
                        continue
                    else:
                        raise ValueError(f'File does not contain {key}') from exc

                # Read the schema
                group = store.root[key]
                schemas[subset] = ColumnSchema.from_json(group._v_attrs.metadata)

            # If no data with this prefix is found, report which ones are found in the file
            if len(data) == 0:
                raise ValueError(f'No data available for prefix "{prefix}". '
                                 'Call `BatteryDataset.inspect_hdf` to gather a list of available prefixes.')

            # Read out the battery metadata
            metadata = BatteryMetadata.model_validate_json(store.root._v_attrs.metadata)
        finally:
            if not is_store:
                store.close()

        return cls(**data, metadata=metadata, schemas=schemas)

    @classmethod
    def all_cells_from_hdf(cls, path: Union[str, Path], subsets: Optional[Collection[str]] = None) -> Iterator[Tuple[str, 'BatteryDataset']]:
        """Iterate over all cells in an HDF file

        Args:
            path: Path to the HDF file
            subsets : Which subsets of data to read from the data file (e.g., raw_data, cycle_stats)
        Yields:
            - Name of the cell
            - Cell data
        """

        # Start by gathering all names of the cells
        _, names = cls.inspect_hdf(path)

        with HDFStore(path, mode='r') as fp:  # Only open once
            for name in names:
                yield name, cls.from_hdf(fp, prefix=name, subsets=subsets)

    @staticmethod
    def inspect_hdf(path_or_buf: Union[str, Path, HDFStore]) -> tuple[BatteryMetadata, Set[Optional[str]]]:
        """Extract the battery data and the prefixes of cells contained within an HDF5 file

        Args:
            path_or_buf: Path to the HDF5 file, or HDFStore object
        Returns:
            - Metadata from this file
            - List of names of batteries stored within the file
        """

        # Get the metadata and list of keys
        if isinstance(path_or_buf, (str, Path)):
            with h5py.File(path_or_buf, 'r') as f:
                metadata = BatteryMetadata.model_validate_json(f.attrs['metadata'])
                keys = list(f.keys())
        else:
            metadata = BatteryMetadata.model_validate_json(path_or_buf.root._v_attrs.metadata)
            keys = [k[1:] for k in path_or_buf.keys()]  # First char is always "/"

        # Get the names by gathering all names before the "-" in group names
        names = set()
        for key in keys:
            for subset in _default_schemas:
                if key.endswith(subset):
                    name = key[:-len(subset) - 1]
                    if len(name) == 0:
                        names.add(None)  # From the default group
                    else:
                        names.add(name)
        return metadata, names

    @staticmethod
    def get_metadata_from_hdf5(path: Union[str, Path]) -> BatteryMetadata:
        """Get battery metadata from an HDF file without reading the data

        Args:
            path: Path to the HDF5 file

        Returns:
            Metadata from this file
        """

        with h5py.File(path, 'r') as f:
            return BatteryMetadata.model_validate_json(f.attrs['metadata'])

    def to_parquet(self, path: Union[Path, str], overwrite: bool = True, **kwargs) -> Dict[str, Path]:
        """Write battery data to a directory of Parquet files

        Keyword arguments are passed to :func:`~pyarrow.parquet.write_table`.

        Args:
            path: Path in which to write to
            overwrite: Whether to overwrite an existing directory
        Returns:
            Map of the name of the subset to
        """
        # Handle existing paths
        path = Path(path)
        if path.exists():
            if not overwrite:
                raise ValueError(f'Path already exists and overwrite is disabled: {path}')
            logger.info(f'Deleting existing directory at {path}')
            shutil.rmtree(path)

        # Make the output directory, then write each Parquet file
        path.mkdir(parents=True, exist_ok=False)
        my_metadata = {
            'battery_metadata': self.metadata.model_dump_json(exclude_none=True),
            'write_date': datetime.now().isoformat()
        }
        written = {}
        for key, schema in self.schemas.items():
            if (data := getattr(self, key)) is None:
                continue
            # Put the metadata for the battery and this specific table into the table's schema in the FileMetaData
            data_path = path / f'{key}.parquet'
            my_metadata['table_metadata'] = schema.model_dump_json()
            table = Table.from_pandas(data, preserve_index=False)
            new_schema = table.schema.with_metadata({**my_metadata, **table.schema.metadata})
            table = table.cast(new_schema)
            pq.write_table(table, where=data_path, **kwargs)

            written[key] = data_path
        return written

    @classmethod
    def from_parquet(cls, path: Union[str, Path], subsets: Optional[Collection[str]] = None):
        """Read the battery data from an HDF file

        Args:
            path: Path to a directory containing parquet files for a specific batter
            subsets: Which subsets of data to read from the data file (e.g., raw_data, cycle_stats)
        """

        # Find the parquet files, if no specification is listed
        path = Path(path)
        if subsets is None:
            subsets = [p.with_suffix('').name for p in path.glob('*.parquet')]

        if len(subsets) == 0:
            raise ValueError(f'No data available for {path}')

        # Load each subset
        metadata = None
        data = {}
        schemas = {}
        for subset in subsets:
            data_path = path / f'{subset}.parquet'
            table = pq.read_table(data_path)

            # Load or check the metadata
            if b'battery_metadata' not in table.schema.metadata:
                warnings.warn(f'Battery metadata not found in {data_path}')
            else:
                # Load the metadata for the whole cell
                my_metadata = table.schema.metadata[b'battery_metadata']
                if metadata is None:
                    metadata = my_metadata
                elif my_metadata != metadata:
                    warnings.warn(f'Battery data different for files in {path}')

            # Load the batdata schema for the table
            if b'table_metadata' not in table.schema.metadata:
                warnings.warn(f'Column schema not found in {data_path}')
            schemas[subset] = ColumnSchema.from_json(table.schema.metadata[b'table_metadata'])

            # Read it to a dataframe
            data[subset] = table.to_pandas()

        return cls(
            metadata=BatteryMetadata.model_validate_json(metadata),
            schemas=schemas,
            **data
        )

    @staticmethod
    def inspect_parquet(path: Union[str, Path]) -> BatteryMetadata:
        """Get battery metadata from a directory of parquet files without reading them

        Args:
            path: Path to the directory of Parquet files

        Returns:
            Metadata from the files
        """

        # Get a parquet file
        path = Path(path)
        if path.is_file():
            pq_path = path
        else:
            pq_path = next(path.glob('*.parquet'), None)
            if pq_path is None:
                raise ValueError(f'No parquet files in {path}')

        # Read the metadata from the schema
        schema = pq.read_schema(pq_path)
        if b'battery_metadata' not in schema.metadata:
            raise ValueError(f'No metadata in {pq_path}')
        return BatteryMetadata.model_validate_json(schema.metadata[b'battery_metadata'])