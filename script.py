import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
import psycopg2
import numpy as np
import os
import csv

print("Initializing project directories...")
os.makedirs('data', exist_ok=True)

uri = 'https://en.wikipedia.org/wiki/List_of_largest_universities_and_university_networks_by_enrollment'
print(f"Fetching data from: {uri}")
response = requests.get(uri)

print("🥣 Parsing HTML content...")
soup = BeautifulSoup(response.content, 'html.parser')

print("🔍 Extracting the target table...")
table = soup.find('table', attrs={'class': 'wikitable sortable'})
trs = table.findAll('tr')

print("📋 Extracting column headers...")
column = []
for item in trs[0].findAll('th'):
    text = item.text.strip()
    print(f"🧩 Column: {text}")
    column.append(text)

columns = list(map(lambda x: x.text.strip(), trs[0].findAll('th')))
columns[-1] = 'Link'
rows = trs[1:]

print(f"📦 Extracting {len(rows)} data rows...")

def extract_rows(tr):
    row_soup_list = tr.findAll('td')
    row = list(map(lambda x: x.text.strip(), row_soup_list))
    link = row_soup_list[1].a.attrs['href'].lstrip('/')
    row[-1] = f"https://en.wikipedia.org/{link}" if link else None
    return row

print("🔁 Extracting first row as a preview...")
print(extract_rows(rows[0]))

data = list(map(extract_rows, rows))

print("💾 Writing raw data to JSON and CSV...")
file_name = 'universities.json'
with open(file_name, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4)

df = pd.DataFrame(data=data, columns=columns)
df.to_csv('universities.csv', index=False)

print("📊 Reading and transforming the dataset...")
df = pd.read_csv('universities.csv')

df.rename(columns={'Distance/In-Person[a]': 'learning_mode'}, inplace=True)

df['learning_modes'] = np.where(
    (df['learning_mode'].str.lower().str.contains('distance')) &
    (df['learning_mode'].str.lower().str.contains('in-person')),
    ['Distance/In-Person'],
    df['learning_mode']
)

df['YearFounded'] = df['Founded'].str.extract(r'(\d{4})')
df['YearFounded'] = df['YearFounded'].astype(int)

df['enrollment'] = df['Enrollment'].str.extract(r'([\d,]+)')
df['enrollment'] = df['enrollment'].str.replace(',', '', regex=False).astype(int)

df.drop(['Founded', 'Enrollment', 'learning_mode'], axis=1, inplace=True)
df.rename(columns={'learning_modes': 'Learning_modes', 'enrollment': 'Enrollment'}, inplace=True)

print("📐 Normalizing dimension tables...")

# Continent
dim_continent = df[['Continent']].drop_duplicates().reset_index(drop=True)
dim_continent['continent_id'] = dim_continent.index + 1
df = df.merge(dim_continent, on='Continent', how='left')

# Affiliation
dim_affiliation = df[['Affiliation']].drop_duplicates().reset_index(drop=True)
dim_affiliation['affiliation_id'] = dim_affiliation.index + 1
df = df.merge(dim_affiliation, on='Affiliation', how='left')

# Learning Modes
dim_learning_modes = df[['Learning_modes']].drop_duplicates().reset_index(drop=True)
dim_learning_modes['learning_modes_id'] = dim_learning_modes.index + 1
df = df.merge(dim_learning_modes, on='Learning_modes', how='left')

# Location
dim_location = df[['Location']].drop_duplicates().reset_index(drop=True)
dim_location['location_id'] = dim_location.index + 1
df = df.merge(dim_location, on='Location', how='left')

# Institution
dim_institution = df[['Institution', 'YearFounded', 'Link']].drop_duplicates().reset_index(drop=True)
dim_institution['institution_id'] = dim_institution.index + 1
df = df.merge(dim_institution, on=['Institution', 'YearFounded', 'Link'], how='left')

print("📐 Creating fact_universities table...")
fact_univerisities = df[['Rank', 'Enrollment', 'continent_id', 'affiliation_id', 'learning_modes_id', 'location_id', 'institution_id']]

print("💾 Writing tables to CSV files...")
fact_univerisities.to_csv('data/fact_univerisities.csv', index=False)
dim_institution.to_csv('data/institution.csv', index=False)
dim_location.to_csv('data/location.csv', index=False)
dim_learning_modes.to_csv('data/learning_modes.csv', index=False)
dim_affiliation.to_csv('data/affiliation.csv', index=False)
dim_continent.to_csv('data/continent.csv', index=False)


# DATABASE
print("🧩 Preparing to create schema and tables in PostgreSQL...")

def get_connection():
    conn = psycopg2.connect(
        host='localhost',
        database='wiki_uni_db',
        user='postgres',
        password='chichi'
    )
    return conn

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    create_schema = '''
        CREATE SCHEMA IF NOT EXISTS wiki;
    '''

    create_tables_sql = '''
        DROP TABLE IF EXISTS wiki.fact_universities CASCADE;
        DROP TABLE IF EXISTS wiki.institution_dim CASCADE;
        DROP TABLE IF EXISTS wiki.affiliation_dim CASCADE;
        DROP TABLE IF EXISTS wiki.location_dim CASCADE;
        DROP TABLE IF EXISTS wiki.learning_modes_dim CASCADE;
        DROP TABLE IF EXISTS wiki.continents_dim CASCADE;

        CREATE TABLE wiki.institution_dim (
            Institution TEXT,
            YearFounded INT,
            Link TEXT,
            institution_id INT PRIMARY KEY
        );

        CREATE TABLE wiki.affiliation_dim (
            Affiliation TEXT,
            affiliation_id INT PRIMARY KEY
        );

        CREATE TABLE wiki.location_dim (
            Location TEXT,
            location_id INT PRIMARY KEY
        );

        CREATE TABLE wiki.learning_modes_dim (
            Learning_modes TEXT,
            learning_modes_id INT PRIMARY KEY
        );

        CREATE TABLE wiki.continents_dim (
            Continent TEXT,
            continent_id INT PRIMARY KEY
        );

        CREATE TABLE wiki.fact_universities (
            Rank INT,
            Enrollment INT,
            continent_id INT REFERENCES wiki.continents_dim(continent_id),
            affiliation_id INT REFERENCES wiki.affiliation_dim(affiliation_id),
            learning_modes_id INT REFERENCES wiki.learning_modes_dim(learning_modes_id),
            location_id INT REFERENCES wiki.location_dim(location_id),
            institution_id INT REFERENCES wiki.institution_dim(institution_id)
        );
    '''

    try:
        print("📁 Creating schema...")
        cursor.execute(create_schema)

        print("📁 Creating tables...")
        cursor.execute(create_tables_sql)

        conn.commit()
        print("✅ All tables created successfully.")

    except Exception as e:
        print(f'❌ Error during table creation: {e}')

    finally:
        cursor.close()
        conn.close()
        print("🔒 Database connection closed.")

create_tables()


# INSERTING DATA

def load_data(csv_file_path, table_name, column_names):
    print(f"⬇️  Loading data into {table_name}...")
    conn = get_connection()
    cursor = conn.cursor()
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                placeholder = ', '.join(['%s'] * len(row))
                query = f'INSERT INTO {table_name} ({", ".join(column_names)}) VALUES({placeholder});'
                cursor.execute(query, row)
        conn.commit()
        print(f"✅ Data loaded into {table_name}")
    except Exception as e:
        print(f"❌ Error loading data into {table_name}: {e}")
    finally:
        cursor.close()
        conn.close()
        print("🔒 Connection closed.\n")

# Load all tables
load_data('data/affiliation.csv', 'wiki.affiliation_dim', ['Affiliation', 'affiliation_id'])
load_data('data/continent.csv', 'wiki.continents_dim', ['Continent', 'continent_id'])
load_data('data/institution.csv', 'wiki.institution_dim', ['Institution', 'YearFounded', 'Link', 'institution_id'])
load_data('data/learning_modes.csv', 'wiki.learning_modes_dim', ['Learning_modes', 'learning_modes_id'])
load_data('data/location.csv', 'wiki.location_dim', ['Location', 'location_id'])
load_data('data/fact_univerisities.csv', 'wiki.fact_universities', [
    'Rank', 'Enrollment', 'continent_id', 'affiliation_id',
    'learning_modes_id', 'location_id', 'institution_id'
])