import csv

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 1. Laden des Datensatzes mit der csv-Bibliothek (mit alternativer Codierung)
def load_data(filename):
    data = []
    with open(filename, mode='r', encoding='ISO-8859-1') as file:  # Versuche 'ISO-8859-1' oder 'latin1'
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            data.append(row)
    return data


# 2. Vorverarbeitung der Daten
def preprocess_data(data):
    # Umwandeln der Daten in ein besser verwendbares Format
    # Entferne alle Zeilen ohne CustomerID
    data = [row for row in data if row['CustomerID'] != '']

    # Erstelle zusätzliche Features für das Clustering
    # Berechne den Gesamtpreis für jedes Produkt (Quantity * UnitPrice)
    for row in data:
        # Ersetze das Komma durch einen Punkt und konvertiere dann zu float
        row['UnitPrice'] = row['UnitPrice'].replace(',', '.')
        row['Quantity'] = row['Quantity'].replace(',', '.')

        try:
            row['TotalPrice'] = float(row['Quantity']) * float(row['UnitPrice'])
        except ValueError:
            print(f"Fehler beim Umwandeln von 'Quantity' oder 'UnitPrice' für Zeile: {row}")
            continue

    # Aggregiere nach CustomerID und berechne:
    # - Gesamtpreis (TotalPrice)
    # - Gesamtanzahl der Bestellungen pro Kunde
    # - Anzahl der verschiedenen Produkte, die ein Kunde gekauft hat
    customer_data = {}
    for row in data:
        customer_id = row['CustomerID']
        if customer_id not in customer_data:
            customer_data[customer_id] = {
                'total_spent': 0,
                'total_purchases': 0,
                'unique_products': set()  # Wir nutzen ein Set, um doppelte Produkte zu vermeiden
            }

        customer_data[customer_id]['total_spent'] += row['TotalPrice']
        customer_data[customer_id]['total_purchases'] += 1
        customer_data[customer_id]['unique_products'].add(row['StockCode'])

    # Umwandeln der aggregierten Daten in eine Liste für die Clusteranalyse
    aggregated_data = []
    for customer_id, data in customer_data.items():
        aggregated_data.append({
            'CustomerID': customer_id,
            'total_spent': data['total_spent'],
            'total_purchases': data['total_purchases'],
            'unique_products': len(data['unique_products'])  # Anzahl einzigartiger Produkte
        })

    return aggregated_data


# 3. Clustering mit KMeans
def kmeans_clustering(data, k):
    # Extrahiere die Merkmale, die für das Clustering verwendet werden
    features = [[d['total_spent'], d['total_purchases'], d['unique_products']] for d in data]

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features)

    # Füge die Clusterzuordnung zu den Daten hinzu
    for i, cluster in enumerate(clusters):
        data[i]['Cluster'] = cluster

    return data, kmeans


# 4. Monitoring der Clustering-Ergebnisse
def monitor_clustering_results_with_range(data):
    print("Clustering Results with Range:")
    for i in range(max(d['Cluster'] for d in data) + 1):
        cluster_data = [d for d in data if d['Cluster'] == i]
        print(f"Cluster {i + 1}:")
        print(f"  Number of customers: {len(cluster_data)}")
        print(f"  Total spent: {sum(d['total_spent'] for d in cluster_data)}")
        print(f"  Total purchases: {sum(d['total_purchases'] for d in cluster_data)}")
        print(f"  Unique products: {sum(d['unique_products'] for d in cluster_data)}")
        print()


# 5. Visualisierung der Cluster
def plot_clusters(data, kmeans):
    total_spent = [d['total_spent'] for d in data]
    total_purchases = [d['total_purchases'] for d in data]
    cluster_labels = [d['Cluster'] for d in data]

    # Verwende eine Color Map für die Clusterfarben
    cmap = plt.get_cmap('viridis', kmeans.n_clusters)

    # Scatter-Plot der Cluster, und speichere das mappable Objekt
    scatter = plt.scatter(total_spent, total_purchases, c=cluster_labels, cmap=cmap)

    # Berechnen der Min/Max-Werte für jedes Cluster und zeichnen der Bereiche
    for i in range(kmeans.n_clusters):
        cluster_data = [d for d in data if d['Cluster'] == i]

        # Berechnung der Min- und Max-Werte für jedes Cluster
        min_spent = min(d['total_spent'] for d in cluster_data)
        max_spent = max(d['total_spent'] for d in cluster_data)
        min_purchases = min(d['total_purchases'] for d in cluster_data)
        max_purchases = max(d['total_purchases'] for d in cluster_data)

        # Clusterfarbe für Rechteck auswählen
        cluster_color = cmap(i / kmeans.n_clusters)  # Erhalte den Farbwert für das Cluster

        # Zeichnen eines Rechtecks für das Cluster-Bereich
        plt.gca().add_patch(
            plt.Rectangle(
                (min_spent, min_purchases),
                max_spent - min_spent,
                max_purchases - min_purchases,
                color=cluster_color,
                alpha=0.3
            )
        )

    # Füge eine colorbar hinzu, die zu den Clusterfarben passt
    plt.colorbar(scatter, label='Cluster')

    # Legende hinzufügen
    handles, labels = scatter.legend_elements()
    plt.legend(handles, labels, title="Cluster")

    plt.xlabel('Total Spent')
    plt.ylabel('Total Purchases')
    plt.title('Customer Clusters')
    plt.show()


# Beispielausführung:
# 1. Datensatz laden
filename = 'resources/customer_data.csv'  # Der Pfad zu deinem CSV-Datensatz
data = load_data(filename)

# 2. Daten vorverarbeiten
customer_data = preprocess_data(data)

# 3. Clustering durchführen (z.B. 5 Cluster)
num_clusters = 5
clustered_data, kmeans_model = kmeans_clustering(customer_data, num_clusters)

# 4. Clustering-Ergebnisse anzeigen
monitor_clustering_results_with_range(clustered_data)

# 5. Cluster visualisieren
plot_clusters(clustered_data, kmeans_model)
