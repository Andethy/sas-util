from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox
from typing import Union, Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame
import numpy as np
from numpy import generic

from constants import EVAL_KEYS_A
from file import JsonFileIO


class PreviewApp:
    track_data: list[list[Any]]

    def __init__(self, window):
        self.track_data = [[]] * 5
        self.root_path = '../resources/study'
        self.data = JsonFileIO(self.root_path + '/results.json').get_entries()

        self.window = window

        window.title('PreviewApp')

        # Use frames for better organization
        entry_frame = tk.Frame(window)
        entry_frame.pack(pady=10)

        stats_frame = tk.Frame(window)
        stats_frame.pack(pady=10)

        graph_frame = tk.Frame(window)
        graph_frame.pack(pady=10)

        control_frame = tk.Frame(window)
        control_frame.pack(pady=10)

        # Song ID Entry
        self.label_song_id = tk.Label(entry_frame, text="Enter Song ID:")
        self.label_song_id.grid(row=0, column=0, padx=5, pady=5)
        self.entry_song_id = tk.Entry(entry_frame)
        self.entry_song_id.grid(row=0, column=1, padx=5, pady=5)

        # Submit Button
        self.submit_button = tk.Button(entry_frame, text="Search", command=self.search_song)
        self.submit_button.grid(row=0, column=2, padx=5, pady=5)

        # Output Text Box for Stats Data
        self.stats_label = tk.Label(stats_frame, text="Song Stats:")
        self.stats_label.pack()
        self.stats_data = tk.Text(stats_frame, height=25, width=50)
        self.stats_data.pack()

        # Graph for Visual Data (using matplotlib)
        self.figure = plt.Figure(figsize=(6.5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas.get_tk_widget().pack()

        # Audio Playback Widget
        self.play_button = tk.Button(control_frame, text="Play", command=self.play_audio)
        self.play_button.pack()

        # Initialize pygame for audio playback
        pygame.mixer.init()
        self.audio_file = ''  # Placeholder for audio file path

    def search_song(self):
        song_id = self.entry_song_id.get()
        self.entry_song_id.delete(0, tk.END)
        self.audio_file = self.get_path(song_id)

        self.clear_text()
        self.stats_data.insert(tk.END, 'Search result for track ID: ' + song_id + "\n\n")

        res = self.analyze_data(song_id)
        if not res:
            return

        for key, vals in res.items():
            self.stats_data.insert(tk.END, key + ':\n')
            for field, val in vals.items():
                self.stats_data.insert(tk.END, f'- {field}: {val}\n')

        # Demonstrate plotting dummy data
        self.ax.clear()
        # self.ax.plot([1, 2, 3], [1, 4, 9])  # Example plot
        self.ax.boxplot(self.track_data, labels=EVAL_KEYS_A)
        self.canvas.draw()

    def analyze_data(self, track) -> dict[str, dict[str, Union[Union[str, str, generic, generic], Any]]] | None:

        # Extract data for each field into separate lists for calculations
        fields_data = {field: [] for field in EVAL_KEYS_A}

        print(self.data)
        try:
            data = self.data[track]
        except KeyError:
            self.clear_text()
            self.add_text(f'Song ID {track} not found')
            return None

        for entry in data.values():
            for field, value in entry.items():
                fields_data[field].append(value)
        self.track_data = list(fields_data.values())
        # Calculate stats for each field
        stats = {}
        for field, values in fields_data.items():
            values = np.array(values)
            stats[field] = {  # Min | Q1 | Median | Q3 | Max
                'Quartiles': ' | '.join([str(n) for n in np.percentile(values, [0, 25, 50, 75, 100])]),
                'Mean': np.mean(values),
                'Standard Deviation': np.std(values)
            }
        return stats

    def play_audio(self):
        if self.audio_file:
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()

        else:
            # Display message or load a default audio file
            self.add_text("Error: Song ID not found")

    def get_path(self, song_id):
        file_name = f"{song_id}.mp3"
        for dir_path, dir_names, file_names in os.walk(self.root_path):
            if file_name in file_names:
                return os.path.join(dir_path, file_name)
        return ''

    def add_text(self, text: str):
        self.stats_data.insert(tk.END, text + "\n")

    def clear_text(self):
        try:
            self.stats_data.delete('1.0', tk.END)
        except tk.TclError:
            pass


class AttributeApp:
    def __init__(self, window):
        self.find_play_button = None
        self.next_button = None  # Initialize the "Next" button
        self.root_path = '../resources/study'
        self.data = JsonFileIO(self.root_path + '/summary.json').get_entries()

        self.window = window
        window.title('AttributeMatcherApp')

        self.check_vars = {key: tk.IntVar() for key in EVAL_KEYS_A}
        self.setup_ui()

        pygame.mixer.init()
        self.audio_file = ''
        self.matches = []  # List to store sorted matches
        self.current_match_index = -1  # Index to track the current match

    def setup_ui(self):
        control_frame = tk.Frame(self.window)
        control_frame.pack(pady=10)

        # Checkboxes for attributes
        for idx, (attr, var) in enumerate(self.check_vars.items()):
            tk.Checkbutton(control_frame, text=attr, variable=var).grid(row=idx // 2, column=idx % 2, sticky='w')

        # Find and Play Button
        self.find_play_button = tk.Button(control_frame, text="Find + Play", command=self.find_and_play)
        self.find_play_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Text widget to display track ID
        self.track_display = tk.Text(control_frame, height=1, width=20)
        self.track_display.grid(row=4, column=0, columnspan=2, pady=5)
        self.track_display.insert(tk.END, "Track ID will appear here")

        # Next Button
        self.next_button = tk.Button(control_frame, text="Next Match", command=self.play_next_match)
        self.next_button.grid(row=5, column=0, columnspan=2, pady=5)

    def find_and_play(self):
        # Reset matches and current index
        self.matches = []
        self.current_match_index = -1

        # Gather user choices
        user_choices = {attr: (var.get() * 6 - 3) for attr, var in self.check_vars.items()}

        # Calculate all matches and sort them by distance
        for key, values in self.data.items():
            distance = 0
            for choice, value in user_choices.items():
                mean_key = f"{choice} Mean"
                if mean_key in values:
                    distance += abs(values[mean_key] - value)
            self.matches.append((key, distance))

        self.matches.sort(key=lambda x: x[1])  # Sort by distance

        # Play the closest match
        self.play_next_match()

    def play_next_match(self):
        self.current_match_index += 1
        if self.current_match_index < len(self.matches):
            track_id = self.matches[self.current_match_index][0]
            self.play_audio(track_id)
            self.track_display.delete('1.0', tk.END)
            self.track_display.insert(tk.END, track_id)
        else:
            messagebox.showerror("Error", "No more matches found.")
            self.current_match_index = -1  # Reset the index

    def play_audio(self, track_id):
        self.audio_file = self.get_path(track_id)
        if self.audio_file:
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()
        else:
            messagebox.showerror("Error", "Audio file not found for the selected attributes.")

    def get_path(self, track_id):
        file_name = f"{track_id}.mp3"
        for dir_path, _, file_names in os.walk(self.root_path):
            if file_name in file_names:
                return os.path.join(dir_path, file_name)
        return ''


# if __name__ == '__main__':
#     root = tk.Tk()
#     app = PreviewApp(root)
#     root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttributeApp(root)
    root.mainloop()
