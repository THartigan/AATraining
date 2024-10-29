import json
import pandas as pd
from IPython.display import display
pd.set_option('display.precision', 10)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
# from tensorflow.keras.preprocessing.sequence import pad_sequences

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
    Pad sequences to the same length.

    Parameters:
    sequences (list of lists): List of sequences to pad.
    maxlen (int): Maximum length of all sequences. If None, uses the length of the longest sequence.
    dtype (str): Type of the output sequences.
    padding (str): 'pre' or 'post' - pad either before or after each sequence.
    truncating (str): 'pre' or 'post' - remove values from sequences longer than maxlen either in the beginning or in the end.
    value (float): Padding value.

    Returns:
    numpy array: Padded sequences.
    """
    lengths = [len(s) for s in sequences]
    if maxlen is None:
        maxlen = max(lengths)
    
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    
    x = np.full((len(sequences), maxlen) + sample_shape, value, dtype=dtype)
    
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')
        
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    
    return x
# uuid = "99AAC7DE-4106-43AB-AA80-F5816AB4DB85"

class importData():
    def __init__(self):
        return
        
    def import_data(self, uuid, spacing, sample_frequency, split_length):
        self.split_length = split_length
        self.sample_frequency = sample_frequency
        self.uuid = uuid
        raw_df, timestamps_df = self.import_uuid(uuid)
        trimmed_raw_df, trimmed_timestamps_df = self.trim_dataframes(raw_df, timestamps_df)
        split_data_list, split_timestamps_list = self.get_split_datasets(trimmed_raw_df, trimmed_timestamps_df, spacing, split_length)
        self.split_data_final, self.split_timestamps_final = self.preprocess_data(split_data_list, split_timestamps_list)
        self.d = self.split_data_final
        self.t = self.split_timestamps_final
        return self.d, self.t
        
    def import_uuid(self, uuid):
        
        # Define the file path
        raw_data_filepath = "Data/" + self.uuid + ".json"
        timestamps_filepath = "Data/Timestamps_" + self.uuid + ".json"
        
        raw_df = self.import_data_json(raw_data_filepath)
        timestamps_df = self.import_timestamp_json(timestamps_filepath)
        return (raw_df, timestamps_df)

    def import_data_json(self, filepath):
        # Read the JSON data from the file
        with open(filepath, 'r') as file:
            data = json.load(file)

        # Extract workoutDatas
        workout_datas = data["workoutDatas"]

        # Create a DataFrame
        df = pd.json_normalize(workout_datas)
        return df

    def import_timestamp_json(self, timestamps_filepath):
        # return pd.read_json(timestamps_filepath)
        with open(timestamps_filepath, 'r') as file:
            data = json.load(file)

        # Step 2: Structure the data
        # Create a dictionary to hold structured data
        structured_data = {
            "startTime": data["startTime"],
            "endTime": data["endTime"],
            "groundLeaveTimes": data["groundLeaveTimes"],
            "airLeaveTimes": data["airLeaveTimes"]
        }

        # Step 3: Create the DataFrame
        # If you want to have each time in a separate row, you can normalize the lists
        df_ground = pd.DataFrame(structured_data["groundLeaveTimes"], columns=["groundLeaveTimes"])
        df_air = pd.DataFrame(structured_data["airLeaveTimes"], columns=["airLeaveTimes"])

        # Combine into a single DataFrame if needed
        df = pd.concat([df_ground, df_air], axis=1)

        # Add startTime and endTime to every row if needed
        df["startTime"] = structured_data["startTime"]
        df["endTime"] = structured_data["endTime"]
        return df
    
    def trim_dataframes(self, raw_df, timestamps_df):
        mask = (raw_df['time'] >= timestamps_df['startTime'][0]) * (raw_df['time'] <= timestamps_df['endTime'][0])
        trimmed_raw_df = raw_df[mask]
        trimmed_raw_df["time"] = trimmed_raw_df["time"].apply(lambda x: x - timestamps_df["startTime"][0])
        timestamps_df["airLeaveTimes"] = timestamps_df["airLeaveTimes"].apply(lambda x: x - timestamps_df["startTime"][0])
        self.end_time = timestamps_df["endTime"][0] - timestamps_df["startTime"][0]
        trimmed_raw_df.pop("id")
        trimmed_raw_df.pop("motion.magneticField.x")
        trimmed_raw_df.pop("motion.magneticField.y")
        trimmed_raw_df.pop("motion.magneticField.z")
        trimmed_raw_df.pop("heartRate")
        trimmed_raw_df.pop("workoutType")
        # trimmed_raw_df.pop("motion.rotationMatrix.m11")
        # trimmed_raw_df.pop("motion.rotationMatrix.m12")
        # trimmed_raw_df.pop("motion.rotationMatrix.m13")
        # trimmed_raw_df.pop("motion.rotationMatrix.m21")
        # trimmed_raw_df.pop("motion.rotationMatrix.m22")
        # trimmed_raw_df.pop("motion.rotationMatrix.m23")
        # trimmed_raw_df.pop("motion.rotationMatrix.m31")
        # trimmed_raw_df.pop("motion.rotationMatrix.m32")
        # trimmed_raw_df.pop("motion.rotationMatrix.m33")
        # trimmed_raw_df.pop("motion.pitch")
        # trimmed_raw_df.pop("motion.roll")"motion.yaw", 
        # trimmed_raw_df.pop("motion.yaw")
        self.processing_column_order = ["time", "motion.pitch", "motion.roll", "motion.gravity.x", "motion.gravity.y", "motion.gravity.z", "motion.acceleration.x", "motion.acceleration.y", "motion.acceleration.z"]#, "motion.rotationMatrix.m11", "motion.rotationMatrix.m12", "motion.rotationMatrix.m13", "motion.rotationMatrix.m21", "motion.rotationMatrix.m22", "motion.rotationMatrix.m23", "motion.rotationMatrix.m31", "motion.rotationMatrix.m32", "motion.rotationMatrix.m33"]
        trimmed_raw_df = trimmed_raw_df[self.processing_column_order]
        # print("df")
        # print(trimmed_raw_df)
        return (trimmed_raw_df, timestamps_df)

    def get_split_datasets(self, trimmed_raw_df, timestamps_df, spacing, split_length):
        """split_length is the time of each split in seconds"""
        start_times = np.arange(0, np.ceil(self.end_time)-split_length, spacing)
        # print(start_times)

        datasets = []
        timestamps = []

        sample_frequency = self.sample_frequency
        for start_time in start_times:
            mask = (trimmed_raw_df["time"] > start_time).tolist()
            try: 
                first_survive = mask.index(1)
                # print("found")
                mask = np.array(mask)
                mask[first_survive + split_length*sample_frequency:] = 0
                start_time_dataset = trimmed_raw_df[mask]
                start_time_dataset["time"] = start_time_dataset["time"] - start_time
                if len(start_time_dataset) == split_length*sample_frequency:
                    datasets.append(start_time_dataset)
                    mask = (timestamps_df["airLeaveTimes"] > start_time) * (timestamps_df["airLeaveTimes"] < start_time +split_length)
                    start_time_timestamps = np.array(timestamps_df["airLeaveTimes"].tolist())[mask]
                    start_time_timestamps = start_time_timestamps - start_time
                    timestamps.append(start_time_timestamps)
            except ValueError:
                print("Could not find start time, skipping this dataset")
                pass
            
            
        # print("dataset")
        # print(datasets_10s[0])
        return (datasets, timestamps)

    def preprocess_data(self, datasets, timestamp_datasets):
        evaluated_datasets = []
        for dataset in datasets:
            x = dataset.values.T#.reshape(1000,19)

            evaluated_datasets.append(x)
        # print(np.shape(evaluated_datasets[0]))
        
        max_seq_length = max(len(seq) for seq in timestamp_datasets)
        Y_padded = pad_sequences(timestamp_datasets, maxlen=max_seq_length, padding='post', dtype='float32')
        num_samples = len(evaluated_datasets)
        Y_padded = Y_padded.reshape(num_samples, max_seq_length, 1)
        X = np.array(evaluated_datasets).reshape(num_samples, len(self.processing_column_order), self.sample_frequency*self.split_length, 1)
        # print(X[0])
        return X, Y_padded

# d, t = importData().import_data(uuid="69508C29-2B15-4230-BE32-328E553EC63D", spacing=0.02, sample_frequency=100, split_length=4)
# print("ddddddddd")
# print(d[0])