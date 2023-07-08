
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generatePlot_all(data):
		grouped_locations = data['province'].unique()

		selected_locations = grouped_locations
		selected_labels = ['positive', 'negative', 'neutral']

		custom_colors = ["#f22443", "#fce938",  "#2df74e"]

		# Filter the DataFrame based on the selected locations and labels
		filtered_df = data[data['province'].isin(selected_locations) & data['label'].isin(selected_labels)]

		# Check if there are any rows in the filtered DataFrame
		if filtered_df.empty:
				print("No data available for the selected locations and labels.")
		else:
				# Group the filtered DataFrame by "location" and "label" and count the occurrences
				grouped_df = filtered_df.groupby(["province", "label"]).size().reset_index(name="count")

				# Set the figure size
				plt.figure(figsize=(10, 8))

				# Create a bar plot
				sns.barplot(x="province", y="count", hue="label", data=grouped_df, palette=custom_colors)

				# Set plot labels and title
				plt.xlabel("Location")
				plt.ylabel("Count")

				# Rotate x-axis labels if needed
				plt.xticks(rotation=45)

				for p in plt.gca().patches:
						plt.gca().annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
				# Show the plot

				plot_all = plt.show()
				return plot_all

