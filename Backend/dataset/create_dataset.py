import csv
import random
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Define the mapping of categories to broader subjects
category_mapping = {
    'Mathematics': ['math'],
    'Computer Science': ['cs'],
    'Physics': ['astro-ph', 'cond-mat', 'gr-qc', 'hep', 'math-ph', 'nlin', 'nucl', 'physics', 'quant-ph'],
    'Statistics': ['stat'],
    'Biology': ['q-bio'],
    'Economics': ['econ']
}

desired_size = 100000

# Initialize a dictionary to store the desired distribution of subjects
desired_distribution = {
    'Mathematics': 0.3,
    'Computer Science': 0.35,
    'Physics': 0.18,
    'Statistics': 0.05,
    'Biology': 0.1,
    'Economics': 0.02
}
desired_counts = {subject: int(count * desired_size) for subject, count in desired_distribution.items()}

input_filename = 'arxiv_data.csv'
output_filename = 'dataset.csv'

# Store data rows for each subject
subject_rows = {subject: [] for subject in category_mapping.keys()}

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    text = text.replace('\r', '')
    return text

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = word_tokenize(text)
    text = ' '.join([w for w in words if not w in stop_words])
    return text


with open(input_filename, mode='r') as input_file:
    reader = csv.DictReader(input_file)
    
    # Iterate through each row in the dataset
    for row in reader:
        category = row['Category']
        
        # Find the corresponding subject for the category and store the row
        for subject, category_list in category_mapping.items():
            if any([category.startswith(prefix) for prefix in category_list]):
                subject_rows[subject].append(row)
                break

with open(output_filename, mode='w', newline='') as output_file:
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    # Write the desired number of rows for each subject to the output file
    for subject, count in desired_counts.items():
        rows = random.sample(subject_rows[subject], count)
        for row in rows:
            row['Title'] = clean_text(row['Title'])
            row['Abstract'] = remove_stopwords(clean_text(row['Abstract']))
            writer.writerow(row)

# Print the results
print("Counts for each subject:")
for subject, count in desired_counts.items():
    print(f"{subject}: {count}")
