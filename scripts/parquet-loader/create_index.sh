curl -X PUT "https://localhost:9200/jvector-index?pretty" --insecure -H 'Content-Type: application/json' -d'
{ 
  "settings": { 
    "index": { 
      "knn": true, 
      "refresh_interval":  -1, 
      "number_of_replicas": 0, 
      "number_of_shards": 1, 
      "merge": { 
         "policy": { 
            "max_merged_segment": "50g" 
          } 
       } 
    } 
  }, 
  "mappings": { 
    "_source": { 
       "excludes": ["embeddings", "chunk_id"], 
        "recovery_source_excludes": ["embeddings", "chunk_id"] 
     }, 
    "properties": { 
      "chunk_id": {"type": "long"}, 
      "embeddings": { 
        "type": "knn_vector", 
        "method": { 
          "name": "disk_ann", 
          "space_type": "l2", 
          "engine": "jvector", 
          "parameters": { 
            "m": 32, 
            "ef_construction": 200, 
            "advanced.num_pq_subspaces": 48 
          } 
        }, 
        "dimension": 384 
      } 
    } 
  } 
}' -u admin
