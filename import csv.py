import csv

data = [
    ["Name", "Age", "City"],
    ["Alice", 24, "Delhi"],
    ["Bob", 29, "Mumbai"],
    ["Charlie", 22, "Bangalore"]
]

with open("sample_data.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerows(data)

print("CSV file created successfully!")
