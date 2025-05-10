############## Demo script for jVector k-NN ##############

# 1. Create an index with knn_vector mapping
curl -X PUT "localhost:9200/jvector-index?pretty" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "index": {
      "knn": true
    }
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "method": {
          "name": "disk_ann",
          "space_type": "l2",
          "engine": "jvector",
          "parameters": {}
        },
        "dimension": 4
      }
    }
  }
}'

# 2. Index 5 documents document with a vector
curl -X PUT "localhost:9200/jvector-index/_doc/1?pretty" -H 'Content-Type: application/json' -d'
{
  "my_vector": [1.0, 2.0, 3.0, 4.0]
}'
curl -X PUT "localhost:9200/jvector-index/_doc/2?pretty" -H 'Content-Type: application/json' -d'
{
  "my_vector": [5.0, 6.0, 7.0, 8.0]
}'
curl -X PUT "localhost:9200/jvector-index/_doc/3?pretty" -H 'Content-Type: application/json' -d'
{
  "my_vector": [9.0, 10.0, 11.0, 12.0]
}'
curl -X PUT "localhost:9200/jvector-index/_doc/4?pretty" -H 'Content-Type: application/json' -d'
{
  "my_vector": [13.0, 14.0, 15.0, 16.0]
}'

# 3. Search for the nearest neighbor of a vector
curl -X GET "localhost:9200/jvector-index/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "size": 1,
  "query": {
    "knn": {
      "my_vector": {
        "vector": [1.0, 2.0, 3.0, 4.0],
        "k": 3
      }
    }
  }
}'



# 4. Get JVector stats after query
curl -X GET "localhost:9200/_plugins/_knn/stats?pretty&stat=knn_query_visited_nodes,knn_query_expanded_nodes,knn_query_expanded_base_layer_nodes" -H 'Content-Type: application/json'