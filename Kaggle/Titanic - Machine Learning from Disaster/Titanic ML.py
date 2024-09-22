# Importo le librerie necessarie:
import pandas as pd  # Importa la libreria pandas e la rinomina come 'pd'
from sklearn.model_selection import train_test_split  # Importa la funzione train_test_split da scikit-learn
from sklearn.ensemble import RandomForestClassifier  # Importa il classificatore RandomForest da scikit-learn
from sklearn.metrics import accuracy_score, classification_report  # Importa le metriche di valutazione:

# accuracy_score e classification_report

# Carico i dati di addestramento
train_data = pd.read_csv('/Users/massimocaramanna/Documents/Data_Science-Kaggle_Competitions/titanic/train.csv')
# Legge i dati di addestramento da un file CSV e li memorizza in un DataFrame chiamato train_data

# Visualizzo le prime righe del dataset
print(train_data.head())  # Stampare le prime righe del DataFrame train_data per visualizzare un'anteprima dei dati

# Visualizzo le informazioni sul dataset
print(train_data.info())  # Stampare informazioni sul DataFrame train_data per ottenere dettagli
# sulla struttura dei dati

# Elimino la colonna 'Embarked'
train_data.drop('Embarked', axis=1, inplace=True)

# Definisco le variabili indipendenti (features) e dipendenti (target)
# Creo un nuovo DataFrame (X) escludendo la colonna 'Survived', che è la variabile target
X = train_data.drop('Survived', axis=1).copy()  # Features
# Controlla se ci sono colonne con dati non numerici
non_numeric_columns = X.select_dtypes(exclude='number').columns
print(f'Colonnes with non-numeric data: {non_numeric_columns}')

# Gestisci le colonne con dati non numerici (se presenti)
for column in non_numeric_columns:
    X[column] = pd.to_numeric(X[column], errors='coerce')

# Rimuovi eventuali righe con valori mancanti dopo la conversione
X = X.dropna()
# Creare una Serie (y) contenente solo la colonna 'Survived', che è la variabile target
y = train_data['Survived'].copy()  # Target

# Converto la colonna 'Sex' in una variabile binaria
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})


# Estrarre 'Title' dalla colonna 'Name'
def extract_title(name):
    titles = name.split(', ')
    if len(titles) > 1:
        title = titles[1].split('.')[0].strip()
        return title
    else:
        return ''


X['Title'] = X['Name'].apply(extract_title)

# Identifico le coppie sposate
married_couples = X[X['SibSp'] > 0]

# Calcolo l'età media per ciascuna coppia sposata
age_map_couples = married_couples.groupby(['Title', 'SibSp'])['Age'].mean().to_dict()


# Assegno l'età media ai dati mancanti solo per i coniugi
def fill_age(row):
    if pd.isnull(row['Age']) and row['SibSp'] > 0 and row['Title'] in age_map_couples:
        return age_map_couples[row['Title']]
    else:
        return row['Age']


X['Age'] = X.apply(fill_age, axis=1)

# Elimino la colonna 'Title' poiché non serve più
X.drop('Title', axis=1, inplace=True)

# Mostra le statistiche descrittive della colonna 'Fare' per ogni classe di biglietto
fare_by_class = X.groupby('Pclass')['Fare'].describe()
print(fare_by_class)

# Unisco la colonna 'Fare' con 'Cabin' per ottenere una caratteristica combinata
X['Fare_Cabin'] = X['Fare'].astype(str) + '_' + X['Cabin']
# Conto il numero di membri della famiglia sulla base della cabina
X['Family_Size'] = X.groupby('Cabin')['Cabin'].transform('count')
# Elimino la colonna 'Fare' poiché ora abbiamo una caratteristica combinata
X.drop('Fare', axis=1, inplace=True)

print("Dimensioni di X:", X.shape)
print("Dimensioni di y:", y.shape)

# Divido il dataset in set di addestramento (training) e di test, utilizzando il 20% dei dati per il test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # random_state è impostato
# a 42 per garantire la riproducibilità dei risultati


# Creo un modello di Random Forest Classifier
model = RandomForestClassifier(random_state=42)  # random_state è impostato a 42 per garantire la
# riproducibilità dei risultati


# Addestro il modello sui dati di addestramento
model.fit(X_train, y_train)

# Effettuo le predizioni sui dati di test
y_pred = model.predict(X_test)

# Valuto l'accuratezza del modello
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza del modello: {accuracy}')

# Visualizzo un report dettagliato delle prestazioni
report = classification_report(y_test, y_pred)
print('Report delle prestazioni:\n', report)

# Visualizzo le prime righe del dataset aggiornato
print('Prime righe dataset aggiornato: ' + train_data.head())

# Visualizzo le informazioni sul dataset aggiornato
print('Informazioni dataset aggiornato: ' + train_data.info())
