import collections
import math

import numpy as np

from constants import *
from file import JsonFileIO, CsvFileIO

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# librosa.feature.mfcc
# librosa.onset.onset_detect

class AudioAnalyzer:
    def __init__(self, in_dir, fields):
        self.fields = fields
        self.out_file = JsonFileIO(in_dir, fields=fields)

    def analyze_data(self, in_field, out_field, buckets: int = 4):
        data = self.out_file.get_entries()
        buffer = []
        max_res = 0
        min_res = math.inf
        print(in_field)
        print(self.fields)
        in_loc = self.fields.index(in_field)
        out_loc = self.fields.index(out_field)

        # for dt in data:
        #     temp = dt[in_field]
        #     if temp > max_res:
        #         max_res = dt[in_field]
        #     elif temp < min_res:
        #         min_res = dt[in_field]

        for dt in data:
            dt[out_field] = self.calculate_bucket(buckets, min_res, max_res, dt[in_field])
            buffer.append(list(dt[field] for field in self.fields))
            # print(*tuple(dt[field] for field in self.fields))
        buffer = sorted(buffer, key=lambda x: x[in_loc])
        for n in range(len(buffer)):
            buffer[n][out_loc] = min(int(n // (len(buffer) // buckets)), buckets - 1)
        self.out_file.add_entries(buffer)

    def classify_data(self, buckets, *jsons):
        data = collections.defaultdict(list)
        temp = collections.defaultdict(str)
        for bucket in buckets:
            data[bucket] = []

        for json in jsons:
            file = JsonFileIO(json)
            for curr in file.get_entries():
                temp[curr["Name"]] = temp[curr["Name"]] + str(curr["bucket"])

        for file, key in temp.items():
            data[key].append(file)

        self.out_file.add_entries_dict(data)

    @staticmethod
    def calculate_bucket(buckets: int, lb: float, ub: float, ac: float):
        bucket = 0
        curr = lb
        for n in range(0, buckets):
            curr += (ub - lb) / buckets
            if ac <= curr:
                break
            bucket += 1
        return bucket

    @staticmethod
    def correlate_buckets(json: JsonFileIO, fields: list, buckets: int = 5):
        print(f'Gathering entries @ {json.file_path}')
        data_list = json.get_entries()

        print("Converting entries to DataFrame object")
        data_df = pd.DataFrame(data_list)

        print("Extracting fields to analyze")
        variables = data_df[fields]

        print("Standardizing data based on fields")
        scaler = StandardScaler()
        variables_scaled = scaler.fit_transform(variables)

        print(f'Performing K-means clustering into {buckets} buckets')
        kmeans = KMeans(n_clusters=buckets, random_state=42)
        data_df['bucket'] = kmeans.fit_predict(variables_scaled)

        print("Calculating cluster centroids")
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(centroids, columns=fields)

        print("Assigning buckets as a field in JSON data")
        for i, row in data_df.iterrows():
            data_list[i]['bucket'] = row['bucket']

        print("Writing new field to JSON")
        json.add_entries(data_list)

        # print("Entries with their respective buckets:")
        # for item in data_list:
        #     print(item)

        # PCA for dimensionality reduction to 2D for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(variables_scaled)
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['principal component 1', 'principal component 2'])

        # PCA loadings (eigenvectors)
        loadings = pca.components_.T  # Transpose to align with original variables: rows=variables, columns=components

        # Create a DataFrame of loadings with the original variables
        loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=fields)

        print("PCA Loadings:")
        print(loadings_df)

        # Adding bucket information to the PCA DataFrame
        principal_df['bucket'] = data_df['bucket']

        # Plotting
        fig, ax = plt.subplots()
        colors = ['r', 'g', 'b', 'y', 'c']
        for bucket, color in zip(principal_df['bucket'].unique(), colors):
            indices_to_keep = principal_df['bucket'] == bucket
            ax.scatter(principal_df.loc[indices_to_keep, 'principal component 1'],
                       principal_df.loc[indices_to_keep, 'principal component 2'],
                       c=color,
                       s=50)
        ax.legend(principal_df['bucket'].unique())
        ax.grid()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA of Clustering Results')
        plt.show()

        print("\nAverage values for each bucket (centroids):")
        print(centroids_df)

        print("Analysis complete - done.")


class StudyAnalyzer:

    def __init__(self, fp, op1, op2, op3, init_entries=TRACKS_ARR):
        self.csv = CsvFileIO(fp, active=True)
        self.out = JsonFileIO(op1)
        self.summary = JsonFileIO(op2)
        self.analyses = JsonFileIO(op3)
        self.files = init_entries
        self.data = self.out.get_entries()

    def init_out(self):
        self.data = {entry: {} for entry in self.files}
        self.out.add_entries_dict(self.data)

    def extract_tracks(self):
        """
        Extracts all responses from the file and stores them in an output JSON file.
        """

        for entry in self.csv.data[START_ROW:]:
            values = list(entry.values())
            person = entry[ID_FIELD]
            items: list = entry[SELECTION_FIELD].split(',')
            for n in RATING_RANGE:
                response = {}
                null_count = 0
                for m, key in enumerate(EVAL_KEYS_A):
                    curr = values[n + m]
                    if not curr:
                        null_count += 1
                        curr = 0
                    else:
                        curr = int(curr)
                    response[key] = curr
                if null_count == len(EVAL_KEYS_A):
                    continue
                elif n >= STUDY_PARTITION:
                    response[EVAL_KEYS_A[0]] *= -1
                self.data[TRACKS_ARR[int(items[0]) - 1]][person] = response
                items.pop(0)
        self.out.add_entries_dict(self.data)

    def extract_people(self, json: JsonFileIO):
        people = {}
        for entry in self.csv.data[START_ROW:]:
            info = {}
            for field in PEOPLE_FIELDS:
                info[field] = entry[field]
            people[entry[ID_FIELD]] = info
        json.add_entries_dict(people)

    def summarize(self):
        track_means = {}
        sd_sorted = []
        for keyTrack, valueTrack in self.data.items():
            person_tracks = [0, 0, 0, 0, 0]
            for keyPerson, valuePerson in valueTrack.items():
                track_data = []
                for keyData, valueData in valuePerson.items():
                    track_data.append(valueData)
                person_tracks = np.vstack((person_tracks, track_data))
            average_values = np.mean(person_tracks, axis=0)
            sd_values = np.std(person_tracks, axis=0)
            track_means[keyTrack] = {key + " Mean": round(value, 2) for key, value in zip(EVAL_KEYS_A, average_values)}
            track_means[keyTrack].update({key + " SD": round(value, 2) for key, value in zip(EVAL_KEYS_A, sd_values)})
            track_means[keyTrack]["Entries"] = len(valueTrack)
            sd_sorted.append((abs(track_means[keyTrack]["Danger SD"]), keyTrack))

        sd_sorted.sort()

        analysis_entries = {}

        lowest20 = [{"Track": item[1], "Danger Mean": track_means[item[1]]["Danger Mean"]} for item in sd_sorted[:20]]
        analysis_entries["Lowest 20 Danger SD"] = lowest20

        highest20 = [{"Track": item[1], "Danger Mean": track_means[item[1]]["Danger Mean"]} for item in sd_sorted[-20:]]
        analysis_entries["Highest 20 Danger SD"] = highest20

        self.analyses.add_entries_dict(analysis_entries, reset=False)

        self.summary.add_entries_dict(track_means)

        danger = []
        urgency = []
        rof = []
        collab = []
        approach = []

        for track_key, tv in track_means.items():
            tv = tuple(tv.values())
            danger.append(tv[0])
            urgency.append(tv[1])
            rof.append(tv[2])
            collab.append(tv[3])
            approach.append(tv[4])

        plt.boxplot([danger, urgency, rof, collab, approach])
        plt.xticks([1, 2, 3, 4, 5], EVAL_KEYS_A, minor=True, rotation=45)
        plt.xlabel(' --------- '.join(EVAL_KEYS_A))

        # Toggle boxplot
        # plt.show()

        """
        # Extracting data
        categories = ['Danger', 'Urgency', 'Risk of Failure', 'Collaboration', 'Approachable']
        category_data = {cat: [] for cat in categories}
        keys = []

        for key, values in track_means.items():
            keys.append(key)
            for i, cat in enumerate(categories):
                category_data[cat].append(values[i])

        # The number of key groups
        n_groups = len(self.data)

        # Create bar chart
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.15
        opacity = 0.8

        for i, cat in enumerate(categories):
            plt.bar(index + i * bar_width, category_data[cat], bar_width, alpha=opacity, label=cat)

        # Adding labels, title, and legend
        plt.xlabel('Keys')
        plt.ylabel('Scores')
        plt.title('Scores by category and key')
        plt.xticks(index + bar_width, keys, rotation='vertical')
        plt.legend()

        # Show plot
        plt.tight_layout()
        plt.show()
        """

    def plot_file(self, file):
        # TODO: validate results.json -> boxplot for ONE audio file across all participants
        pass


# analyzer = Analyzer(JSON_ONSET_PATH, fields=ONSET_FIELDS)
# analyzer.analyze_data('Onsets', 'BUCKET', 5)
# analyzer = AudioAnalyzer(JSON_BUCKETS_PATH, fields=OUTPUT_BUCKETS)
# analyzer.correlate_buckets(JsonFileIO(JSON_FEATURES_PATH), list(FEATURES_FIELDS_SMALL), 10)
# analyzer.classify_data(OUTPUT_BUCKETS, JSON_FEATURES_PATH)
# ppl = JsonFileIO('../resources/study/people.json')
# analyzer.extract_people(ppl)
# ppl.numerize_entries(**PEOPLE_TYPES)
if __name__ == '__main__':
    analyzer = StudyAnalyzer('../resources/study/study.csv',
                             '../resources/study/results.json',
                             '../resources/study/summary.json',
                             '../resources/study/analysis.json')
    # analyzer.extract_tracks()
    analyzer.summarize()
