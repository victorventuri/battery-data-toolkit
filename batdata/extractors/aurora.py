"""Extractor for Arbin-format files from Aurora Flight Sciences"""
import os
from typing import Union, List, Iterator, Tuple, Optional

import numpy as np
import pandas as pd

from batdata.data import BatteryDataset
from batdata.schemas import BatteryMetadata
from batdata.extractors.base import BatteryDataExtractor
from batdata.schemas.cycling import ChargingState
from batdata.utils import drop_cycles
from batdata.postprocess.tagging import AddMethod, AddSteps, AddSubSteps

class AuroraExtractor(BatteryDataExtractor):
    """Parser for reading from Arbin-format files from Aurora Flight Sciences

    Expects the files to be in Excel (.xlsx) format
    """

    def implementors(self) -> List[str]:
        return ['Venturi, Victor <vventuri@anl.gov>']
    
    def version(self) -> str:
        return '0.0.1'
    
    def identify_files(self, 
                       path: str, 
                       context: dict = None) -> Iterator[Tuple[str]]:
        '''
        Identify all groups of files likely to be compatible with this extractor

        Uses the :meth:`group` function to determine groups of files that should be parsed together.

        Parameters
        ----------
            path: str 
                Root of directory to group together
            context: dict 
                Context about the files

        Returns
        -------
            group: iterable [str]
                Groups of eligible files
        '''

        # Walk through the directories
        for root, dirs, files in os.walk(path):
            # Sort to make sure we take files in order
            dirs.sort()

            # Generate the full paths
            dirs = [os.path.join(root, d) for d in dirs]
            files = [os.path.join(root, f) for f in files]

            # Get any groups from this directory
            for group in self.group(files, dirs, context):
                yield group
    
    def group(self, 
              files: Union[str, List[str]], 
              directories: List[str] = None,
              context: dict = None) -> Iterator[Tuple[str, ...]]:
        for file in files:
            if file.lower().endswith('.xlsx'):
                yield file

    def generate_summary_dataframe(self, 
                                    file: str, 
                                    sheet: str, 
                                    file_number: int = 0,
                                    start_cycle: int = 0,
                                    start_time: float = 0) \
                                        -> Tuple[pd.DataFrame, int, float]:
        '''
        Generate pandas.DataFrame containing cycle stats in file, at 
        specified sheet.

        Parameters
        ----------
        file: str
            Path to the Excel file 
        sheet: str
            Name of sheet where data is stored 
        file_number: int
            Number of file, in case the test is spread across multiple files 
        start_cycle: int or np.int64
            Index to use for the first cycle, in case test is spread across 
            multiple files
        start_time: float or np.float64
            Test time to use for the start of the test, in case test is spread 
            across multiple files

        Returns
        -------
        df_out: pd.DataFrame
            Dataframe containing the battery data in a standard format
        end_cycle: int
            Index of the final cycle
        end_time: float
            Test time of the last measurement
        '''

        # Read the file and rename the file
        df = pd.read_excel(file, sheet_name = sheet)

        # First, convert Date_time to UNIX timestamps
        unix_timestamps = []
        for datetime in df['Date_Time'].values:
            unix_timestamps.append(pd.Timestamp(datetime).timestamp())
        # convert to numpy
        unix_timestamps = np.array(unix_timestamps)
        # subtract cycle durations to find out start time 
        cycle_starts = unix_timestamps - df['Test Time (s)'].to_numpy() 
        cycle_starts += start_time # considering the start time

        # Create fresh DataFrame
        df_out = pd.DataFrame()

        # Convert column names 
        df_out['cycle_number'] = df['Cycle Index'] + start_cycle - \
                                df['Cycle Index'].min()
        df_out['cycle_start'] = cycle_starts
        df_out['cycle_duration'] = df['Test Time (s)']
        df_out['discharge_capacity'] = df['Discharge Capacity (Ah)']
        df_out['charge_capacity'] = df['Charge Capacity (Ah)']
        df_out['coulomb_efficiency'] = df['Coulombic Efficiency (%)']
        # Remember to convert the energy to Joule
        df_out['discharge_energy'] = df['Discharge Energy (Wh)'].to_numpy() / 3600
        df_out['charge_energy'] = df['Charge Energy (Wh)'].to_numpy() / 3600
        df_out['file_number'] = file_number

        return df_out, \
                df_out['cycle_number'].max(), \
                unix_timestamps.max()       

    def generate_dataframe(self, 
                           file: str, 
                           sheet: str, 
                           file_number: int = 0,
                           start_cycle: int = 0,
                           start_time: float = 0) \
                            -> Tuple[pd.DataFrame, int, float]:
        '''
        Generate pandas.DataFrame containing data in file, at specified sheet.

        Parameters
        ----------
        file: str
            Path to the Excel file 
        sheet: str
            Name of sheet where data is stored 
        file_number: int
            Number of file, in case the test is spread across multiple files 
        start_cycle: int or np.int64
            Index to use for the first cycle, in case test is spread across 
            multiple files
        start_time: float or np.float64
            Test time to use for the start of the test, in case test is spread 
            across multiple files

        Returns
        -------
        df_out: pd.DataFrame
            Dataframe containing the battery data in a standard format
        end_cycle: int
            Index of the final cycle
        end_time: float
            Test time of the last measurement
        '''

        # Read the file and rename the file
        df = pd.read_excel(file, sheet_name = sheet)

        # First, convert Date_time to UNIX timestamps
        unix_timestamps = []
        for datetime in df['Date_Time'].values:
            unix_timestamps.append(pd.Timestamp(datetime).timestamp())
        
        # Now, find columns that contain temperature readings
        temp_cols = [col_name for col_name in df.columns \
                     if 'Temperature' in col_name]
        temp_readings = df[temp_cols].to_numpy()
        # Average them along the rows
        temps = temp_readings.mean(axis = 1)

        # Create fresh DataFrame
        df_out = pd.DataFrame()

        # Convert column names 
        df_out['time'] = unix_timestamps
        df_out['cycle_number'] = df['Cycle Index'] + start_cycle - \
                                df['Cycle Index'].min()
        df_out['cycle_number'] = df_out['cycle_number'].astype('int64')
        df_out['test_time'] = np.array(df['Test Time (s)'] + \
                                start_time, dtype=float)
        df_out['current'] = df['Current (A)']
        df_out['voltage'] = df['Voltage (V)']
        df_out['step_index'] = df['Step Index']
        df_out['temperature'] = temps
        df_out['internal_resistance'] = df['Internal Resistance (Ohm)']
        df_out['file_number'] = file_number  # df_out['cycle_number']*0
        
        # Drop the duplicate rows
        df_out = drop_cycles(df_out)

        # Determine whether the battery is charging or discharging:
        #   0 is rest, 1 is charge, -1 is discharge
        # TODO (wardlt): This function should move to post-processing
        def compute_state(x):
            if abs(x) < self.eps:
                return ChargingState.hold
            return ChargingState.charging if x > 0 else ChargingState.discharging

        df_out['state'] = df_out['current'].apply(compute_state)

        # Determine the method uses to control charging/discharging
        # AddSteps().enhance(df_out) # step_index already in original file!!!
        AddMethod().enhance(df_out)
        AddSubSteps().enhance(df_out)
        return df_out, \
                df_out['cycle_number'].max(), \
                df_out['test_time'].max() 

    def parse_to_dataframe(self, 
                           group: List[str], 
                           info_to_extract: str = 'all', 
                           metadata: Optional[Union[BatteryMetadata, \
                                    dict]] = None) -> BatteryDataset:
        '''
        Parse a set of  files into a Pandas dataframe

        Parameters
        ----------
        group: list of str
            List of files to parse as part of the same test. Ordered 
            sequentially
        info_to_extract: str
            Options are 'raw', 'summary', or 'all'
        metadata: dict, optional
            Metadata for the battery, should adhere to the BatteryMetadata 
            schema

        Returns
        -------
        BatteryDataset
            DataFrame containing the information from all files
        '''

        # Initialize counters for the cycle numbers, etc.. 
        start_cycle = 0
        start_time = 0

        # Read the data for each file
        #  Keep track of the ending index and ending time
        output_dfs = {'raw': [], 'summary': []}

        for file_number, file in enumerate(group):
            # Get relevanty sheet names 
            fileinfo = pd.ExcelFile(file)
            sheet_names = fileinfo.sheet_names
            raw = [name for name in sheet_names if 'Channel' in name]
            summary = [name for name in sheet_names if \
                       'StatisticsByCycle' in name]
            raw_sheet = raw[0]
            summary_sheet = summary[0]
            
            # Process the file
            if info_to_extract != 'raw': # This does not work yet
                df_out, end_cycle, end_time = self.generate_summary_dataframe(
                                                    file, 
                                                    summary_sheet,
                                                    file_number, 
                                                    start_cycle, 
                                                    start_time)
                output_dfs['summary'].append(df_out)
            if info_to_extract != 'summary':
                df_out, end_cycle, end_time = self.generate_dataframe(file, 
                                                                raw_sheet,
                                                                file_number, 
                                                                start_cycle, 
                                                                start_time)
                output_dfs['raw'].append(df_out)
            

            # Increment the start cycle and time to determine starting point of 
            # next file
            start_cycle = end_cycle + 1
            start_time = end_time

        # Combine the data from all files
        if info_to_extract != 'raw': 
            summary_out = pd.concat(output_dfs['summary'], ignore_index=True)
        if info_to_extract != 'summary':
            raw_out = pd.concat(output_dfs['raw'], ignore_index=True)

        # Returns   
        if info_to_extract == 'raw':
            return BatteryDataset(raw_data = raw_out, 
                                  metadata = metadata)
        elif info_to_extract == 'summary':
            return BatteryDataset(cycle_stats = summary_out, 
                                  metadata = metadata)
        elif info_to_extract == 'all':
            return BatteryDataset(raw_data = raw_out, 
                                  cycle_stats = summary_out,
                                  metadata = metadata)