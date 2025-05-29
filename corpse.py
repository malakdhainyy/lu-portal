import csv
import os
import re
from collections import defaultdict

def sanitize_filename(name):
    return re.sub(r'\W+', '_', name.strip())

# Load publications from existing CSV
publications_by_researcher = defaultdict(list)
with open("all_university_publications.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        researcher = row.get("Researcher Name", "Unknown")
        publications_by_researcher[researcher].append(row)

# Create output folder
os.makedirs("researcher_corpora", exist_ok=True)

# Save individual researcher files
for researcher, pubs in publications_by_researcher.items():
    filename = sanitize_filename(researcher) + "_publications.csv"
    filepath = os.path.join("researcher_corpora", filename)
    fields = pubs[0].keys()

    with open(filepath, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pub in pubs:
            writer.writerow(pub)

    print(f"✅ Saved {len(pubs)} publications for: {researcher}")



import os

folder_path = "researcher_corpora"

csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
print(f"📄 Number of researcher CSV files: {len(csv_files)}")
