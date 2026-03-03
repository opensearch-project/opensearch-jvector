# Maximal Marginal Relevance (MMR) Search in JVector

## 1. Introduction to MMR

Maximal Marginal Relevance (MMR) is a technique used to balance relevance and diversity in search results. In traditional vector similarity search, results are ranked purely by their similarity to the query vector, which often leads to redundant results that are very similar to each other. MMR addresses this "filter bubble" problem by selecting results that are both relevant to the query and diverse from each other.

## 2. MMR Formula

The MMR algorithm iteratively selects documents from a candidate set by maximizing the following score:

```
MMR = λ × Sim₁(D, Q) - (1 - λ) × max[Sim₂(D, Dᵢ)]
                                    Dᵢ∈S
```

Where:
- **D**: Candidate document being evaluated
- **Q**: Query vector
- **S**: Set of already selected documents
- **λ** (lambda/diversity): Weight parameter in range [0, 1]
  - λ = 0: Maximum diversity (only considers dissimilarity to selected docs)
  - λ = 1: Maximum relevance (standard KNN, ignores diversity)
  - λ = 0.5: Balanced approach (default)
- **Sim₁(D, Q)**: Similarity between candidate document D and query Q
- **Sim₂(D, Dᵢ)**: Similarity between candidate D and already selected document Dᵢ
- **max[Sim₂(D, Dᵢ)]**: Maximum similarity to any already selected document

### Algorithm Steps

1. Start with an empty selected set S
2. Retrieve a candidate set C (oversampled results from KNN search)
3. While |S| < k (desired result count):
   - For each candidate D in C:
     - Calculate relevance: `Sim₁(D, Q)`
     - Calculate max similarity to selected: `max[Sim₂(D, Dᵢ)]` for all Dᵢ in S
     - Compute MMR score: `λ × Sim₁(D, Q) - (1 - λ) × max[Sim₂(D, Dᵢ)]`
   - **Select the document with the highest MMR score** (this is the key selection criterion)
   - Move selected document from C to S
4. Return S as final results

**Important**: At each iteration, the algorithm evaluates all remaining candidates and selects the one that maximizes the MMR score, balancing relevance to the query with diversity from already-selected documents.

## 3. Implementation in JVector

The MMR implementation in JVector uses a two-processor pipeline architecture integrated with OpenSearch's search pipeline framework.

### Architecture Overview

```
User Request → MMRSearchExtBuilder → MMROverSampleProcessor → KNN Search → MMRRerankProcessor → Final Results
```

### Key Components

#### 3.1 MMRSearchExtBuilder

**Location**: [`src/main/java/org/opensearch/knn/search/extension/MMRSearchExtBuilder.java`](../src/main/java/org/opensearch/knn/search/extension/MMRSearchExtBuilder.java)

This is the search extension that users include in their queries to enable MMR. It parses and validates MMR parameters.

**Parameters**:
- `diversity` (float, 0.0-1.0): Controls the diversity weight (default: 0.5)
- `candidates` (integer): Number of candidates to oversample (default: 3 × query size)
- `vector_field_path` (string, optional): Path to the vector field for reranking
- `vector_field_data_type` (string, optional): Data type of vector field (float/byte)
- `vector_field_space_type` (string, optional): Similarity function (l2/cosinesimil/etc.)

**Example Usage**:
```json
{
  "query": {
    "knn": {
      "restaurant_embedding": {
        "vector": [1.0, 1.0, 1.0, 1.0, 1.0],
        "k": 5
      }
    }
  },
  "ext": {
    "mmr": {
      "diversity": 0.5,
      "candidates": 8
    }
  }
}
```

#### 3.2 MMROverSampleProcessor

**Location**: [`src/main/java/org/opensearch/knn/search/processor/mmr/MMROverSampleProcessor.java`](../src/main/java/org/opensearch/knn/search/processor/mmr/MMROverSampleProcessor.java)

A system-generated search request processor that modifies the query before execution.

**Responsibilities**:
1. **Extract MMR Configuration**: Parses the MMR extension from the search request
2. **Compute Candidates**: Determines oversampling size (default: 3 × original query size)
3. **Modify Query Size**: Updates `request.source().size(candidates)` to fetch more results
4. **Preserve Source Context**: Ensures `_source` is enabled to retrieve vectors for reranking
5. **Resolve Vector Field Info**: Determines vector field path, data type, and space type from:
   - User-provided parameters (required for remote indices)
   - Index mappings (for local indices)
6. **Store Context**: Saves [`MMRRerankContext`](../src/main/java/org/opensearch/knn/search/processor/mmr/MMRRerankContext.java) in pipeline context for the rerank processor

**Key Methods**:
- `processRequestAsync()`: Main entry point for request transformation
- `computeCandidatesAndSetRequestSize()`: Calculates and sets oversampling size
- `preserveAndEnableFullSource()`: Ensures vectors are available in response
- `transformQueryForMMR()`: Delegates to query-specific transformers

#### 3.3 MMRRerankProcessor

**Location**: [`src/main/java/org/opensearch/knn/search/processor/mmr/MMRRerankProcessor.java`](../src/main/java/org/opensearch/knn/search/processor/mmr/MMRRerankProcessor.java)

A system-generated search response processor that reranks results using MMR.

**Responsibilities**:
1. **Extract Vectors**: Retrieves vector embeddings from `_source` of each hit
2. **Apply MMR Algorithm**: Iteratively selects documents maximizing MMR score
3. **Rerank Results**: Replaces original hits with MMR-selected subset
4. **Restore Source Context**: Applies original `_source` filtering if needed

**Key Methods**:
- `processResponse()`: Main entry point for response processing
- `extractVectors()`: Extracts embeddings from search hits using [`MMRUtil.extractVectorFromHit()`](../src/main/java/org/opensearch/knn/search/processor/mmr/MMRUtil.java:312)
- `selectHitsWithMMR()`: Implements the core MMR selection algorithm
- `applyFetchSourceFilterIfNeeded()`: Restores user's original `_source` preferences

### Execution Sequence

1. **User submits search request** with MMR extension
2. **Cluster settings validation**: Ensures MMR processors are enabled
3. **MMROverSampleProcessor**:
   - **POST_USER_DEFINED**: This processor runs *after* any user-defined search request processors in the pipeline. This ensures that user customizations are applied first, then MMR oversampling happens as a final transformation before the query executes.
   - Extracts MMR parameters
   - Resolves vector field information
   - Increases query size to `candidates` or 3x by default
   - Enables full `_source` retrieval
   - Stores context in pipeline
4. **KNN Search executes** with oversampled size
5. **MMRRerankProcessor**:
   - **PRE_USER_DEFINED**: This processor runs *before* any user-defined search response processors. Since we oversampled, it's better to rerank and reduce results to the original size before other response processors run, ensuring they work with the final result set.
   - Retrieves MMR context
   - Extracts vectors from hits
   - Applies MMR algorithm
   - Selects top-k diverse results
   - Restores original `_source` filtering
6. **Final results returned** to user

### Enabling MMR Processors

MMR processors must be explicitly enabled in cluster settings:

```bash
curl -X PUT "localhost:9200/_cluster/settings" -H 'Content-Type: application/json' -d '{
  "persistent": {
    "cluster.search.enabled_system_generated_factories": [
      "mmr_over_sample_factory",
      "mmr_rerank_factory"
    ]
  }
}'
```

## 4. MMR Calculation - Restaurant Search Example

Let's walk through a complete MMR search using the dataset from [`scripts/test_mmr_restaurants.sh`](../scripts/test_mmr_restaurants.sh).

### Dataset Overview

The test script indexes 8 restaurants in San Jose with 5-dimensional embeddings:

| ID | Restaurant | Cuisine | Embedding | Rating | Price |
|----|-----------|---------|-----------|--------|-------|
| 1 | Paesano Ristorante Italiano | Italian | [1.0, 1.0, 1.0, 1.0, 1.0] | 4.5 | $$$ |
| 2 | Maggianos Little Italy | Italian | [1.1, 1.1, 1.1, 1.1, 1.1] | 4.3 | $$ |
| 3 | Osteria Coppa | Italian | [1.2, 1.2, 1.2, 1.2, 1.2] | 4.4 | $$$ |
| 4 | Sushi Confidential | Japanese | [2.0, 2.0, 2.0, 2.0, 2.0] | 4.6 | $$$ |
| 5 | Minato Japanese Restaurant | Japanese | [2.1, 2.1, 2.1, 2.1, 2.1] | 4.4 | $$ |
| 6 | La Victoria Taqueria | Mexican | [5.0, 5.0, 5.0, 5.0, 5.0] | 4.2 | $ |
| 7 | Pho Ha Noi | Vietnamese | [0.5, 0.5, 0.5, 0.5, 0.5] | 4.5 | $ |
| 8 | Amber India | Indian | [3.5, 3.5, 3.5, 3.5, 3.5] | 4.3 | $$ |

**Embedding Clusters**:
- **Cluster 1** (Italian): [1.0-1.2] - very similar to each other
- **Cluster 2** (Japanese): [2.0-2.1] - moderately different from Italian
- **Cluster 3** (Mexican): [5.0] - very different
- **Cluster 4** (Vietnamese): [0.5] - different direction
- **Cluster 5** (Indian): [3.5] - another diverse option

### Query Configuration

**Query Vector**: `[1.0, 1.0, 1.0, 1.0, 1.0]` (looking for Italian-style restaurants)

**MMR Parameters**:
- `diversity` (λ): 0.5 (balanced relevance and diversity)
- `candidates`: 8 (all restaurants)
- `size`: 5 (want top 5 results)
- `space_type`: L2 (Euclidean distance)

### Step-by-Step MMR Calculation with λ = 0.5

For MMR calculations, we convert L2 distances to similarity scores using: **Sim = 1 / (1 + distance)**

This ensures:
- Distance = 0 → Similarity = 1.0 (identical vectors)
- Distance = ∞ → Similarity = 0.0 (completely different)

#### Initial KNN Search Results (by L2 distance)

First, the system retrieves 8 candidates ranked by L2 distance to query [1.0, 1.0, 1.0, 1.0, 1.0]:

| Rank | Restaurant | L2 Distance | Similarity Score |
|------|-----------|-------------|------------------|
| 1 | Paesano (ID 1) | 0.0 | 1.000 |
| 2 | Maggianos (ID 2) | 0.224 | 0.817 |
| 3 | Osteria (ID 3) | 0.447 | 0.691 |
| 4 | Pho Ha Noi (ID 7) | 1.118 | 0.472 |
| 5 | Sushi (ID 4) | 2.236 | 0.309 |
| 6 | Minato (ID 5) | 2.460 | 0.289 |
| 7 | Amber India (ID 8) | 5.590 | 0.152 |
| 8 | La Victoria (ID 6) | 8.944 | 0.100 |

#### MMR Iteration 1: Select First Document

For the first selection, there are no previously selected documents (S is empty), so the diversity penalty is zero.

**Why is the diversity penalty zero?**

Using the MMR formula:
```
MMR(D) = (1 - λ) × Sim(D, Q) - λ × max[Sim(D, Dᵢ)]
                                      Dᵢ∈S
```

When S is empty (no documents selected yet):
- There are no documents Dᵢ in S to compare against
- Therefore, `max[Sim(D, Dᵢ)]` over an empty set = 0
- The diversity penalty term `λ × max[Sim(D, Dᵢ)]` = λ × 0 = 0

So the formula simplifies to:
```
MMR(D) = (1 - λ) × Sim(D, Q) - λ × 0
       = (1 - λ) × Sim(D, Q)
       = (1 - 0.5) × Sim(D, Q)
       = 0.5 × Sim(D, Q)
```

This means the first selection is based purely on relevance to the query (scaled by 1-λ).

**Scores** (we select the candidate with the highest MMR score):
- Paesano: 0.5 × 1.000 = **0.500** ← **HIGHEST MMR SCORE - SELECTED**
- Maggianos: 0.5 × 0.817 = 0.409
- Osteria: 0.5 × 0.691 = 0.346
- (others have lower scores)

**Selected Set S**: [Paesano]

#### MMR Iteration 2: Select Second Document

Now we must consider similarity to Paesano [1.0, 1.0, 1.0, 1.0, 1.0].

**Key Insight**: Since the query vector [1.0, 1.0, 1.0, 1.0, 1.0] equals Paesano's embedding, all candidates have the same distance to both the query and Paesano. This creates a special case where we need to look at the actual implementation.

In the actual JVector implementation, the MMR formula is:

```java
double score = (1 - diversity) * candidate.getScore() - diversity * maxSimToSelected;
```

Where:
- `candidate.getScore()` = Original KNN relevance score (from initial search)
- `maxSimToSelected` = Maximum similarity to any already selected document

The KNN scores from the initial search are based on distance to the query. Since Paesano equals the query, similarities to Paesano equal similarities to the query for all candidates.

**MMR Formula Applied** (λ = 0.5):
```
MMR = (1 - 0.5) × KNN_Score - 0.5 × Sim_to_Paesano
    = 0.5 × KNN_Score - 0.5 × Sim_to_Paesano
```

Since Sim_to_Query = Sim_to_Paesano for all candidates (because Query = Paesano):

```
MMR = 0.5 × Sim_to_Query - 0.5 × Sim_to_Query = 0
```

All candidates get MMR score of 0! In this tie-breaking scenario, the implementation selects based on the original KNN ranking order.

**Result**: **Maggianos** is selected (next in KNN ranking) ← **SELECTED**

**Selected Set S**: [Paesano, Maggianos]

**Note**: This is a special edge case where the query exactly matches a document. In real-world scenarios, the query vector typically doesn't exactly match any document, leading to meaningful MMR score differences.

#### MMR Iteration 3: Select Third Document (More Realistic Scenario)

To demonstrate MMR more clearly, let's assume Paesano and Maggianos are now selected. For the remaining candidates, we calculate:

Max similarity to selected {Paesano [1.0], Maggianos [1.1]}:
- Maggianos: 0.978
- Osteria: 0.957  
- Pho Ha Noi: 0.691
- Sushi: 0.309

Similarities to Paesano (using inverse distance):
- Maggianos: 0.817 (very similar)
- Osteria: 0.691 (similar)
- Pho Ha Noi: 0.472 (moderately different)
- Sushi: 0.309 (quite different)

MMR = 0.5 × KNN_score - 0.5 × sim_to_selected:
- **Maggianos**: 0.5 × 0.978 - 0.5 × 0.817 = 0.489 - 0.409 = **0.080**
- **Osteria**: 0.5 × 0.957 - 0.5 × 0.691 = 0.479 - 0.346 = **0.133**
- **Pho Ha Noi**: 0.5 × 0.691 - 0.5 × 0.472 = 0.346 - 0.236 = **0.110**
- **Sushi**: 0.5 × 0.309 - 0.5 × 0.309 = 0.155 - 0.155 = **0.000**

**Osteria** has the highest MMR score (0.133) ← **SELECTED**

**Selected Set S**: [Paesano, Osteria]

#### MMR Iteration 3: Select Third Document

Now we compare to both Paesano and Osteria, taking the maximum similarity.

Similarities to Osteria [1.2]:
- Maggianos [1.1]: dist=0.224, sim=0.817
- Pho Ha Noi [0.5]: dist=1.565, sim=0.390
- Sushi [2.0]: dist=1.789, sim=0.359
- Minato [2.1]: dist=2.013, sim=0.332
- Amber India [3.5]: dist=5.145, sim=0.163
- La Victoria [5.0]: dist=8.500, sim=0.105

Max similarity to selected {Paesano, Osteria}:
- **Maggianos**: max(0.817 to Paesano, 0.817 to Osteria) = 0.817
- **Pho Ha Noi**: max(0.472 to Paesano, 0.390 to Osteria) = 0.472
- **Sushi**: max(0.309 to Paesano, 0.359 to Osteria) = 0.359
- **Minato**: max(0.289 to Paesano, 0.332 to Osteria) = 0.332
- **Amber India**: max(0.152 to Paesano, 0.163 to Osteria) = 0.163
- **La Victoria**: max(0.100 to Paesano, 0.105 to Osteria) = 0.105

MMR Scores:
- **Maggianos**: 0.5 × 0.978 - 0.5 × 0.817 = **0.080**
- **Pho Ha Noi**: 0.5 × 0.691 - 0.5 × 0.472 = **0.110**
- **Sushi**: 0.5 × 0.309 - 0.5 × 0.359 = **-0.025** (negative!)
- **Amber India**: 0.5 × 0.152 - 0.5 × 0.163 = **-0.006** (negative!)
- **La Victoria**: 0.5 × 0.100 - 0.5 × 0.105 = **-0.003** (negative!)

**Pho Ha Noi** has the highest MMR score (0.110) ← **SELECTED**

**Selected Set S**: [Paesano, Osteria, Pho Ha Noi]

#### MMR Iteration 4: Select Fourth Document

Similarities to Pho Ha Noi [0.5]:
- Maggianos [1.1]: dist=1.342, sim=0.427
- Sushi [2.0]: dist=3.354, sim=0.230
- Minato [2.1]: dist=3.578, sim=0.218
- Amber India [3.5]: dist=6.708, sim=0.130
- La Victoria [5.0]: dist=10.062, sim=0.090

Max similarity to selected {Paesano, Osteria, Pho Ha Noi}:
- **Maggianos**: max(0.817, 0.817, 0.427) = 0.817
- **Sushi**: max(0.309, 0.359, 0.230) = 0.359
- **Minato**: max(0.289, 0.332, 0.218) = 0.332
- **Amber India**: max(0.152, 0.163, 0.130) = 0.163
- **La Victoria**: max(0.100, 0.105, 0.090) = 0.105

MMR Scores:
- **Maggianos**: 0.5 × 0.978 - 0.5 × 0.817 = **0.080**
- **Sushi**: 0.5 × 0.309 - 0.5 × 0.359 = **-0.025**
- **Minato**: 0.5 × 0.289 - 0.5 × 0.332 = **-0.022**
- **Amber India**: 0.5 × 0.152 - 0.5 × 0.163 = **-0.006**
- **La Victoria**: 0.5 × 0.100 - 0.5 × 0.105 = **-0.003**

**Maggianos** has the highest MMR score (0.080) ← **SELECTED**

**Selected Set S**: [Paesano, Osteria, Pho Ha Noi, Maggianos]

#### MMR Iteration 5: Select Fifth Document

Max similarity to selected {Paesano, Osteria, Pho Ha Noi, Maggianos}:
- **Sushi**: max(0.309, 0.359, 0.230, 0.309) = 0.359
- **Minato**: max(0.289, 0.332, 0.218, 0.289) = 0.332
- **Amber India**: max(0.152, 0.163, 0.130, 0.152) = 0.163
- **La Victoria**: max(0.100, 0.105, 0.090, 0.100) = 0.105

MMR Scores:
- **Sushi**: 0.5 × 0.309 - 0.5 × 0.359 = **-0.025**
- **Minato**: 0.5 × 0.289 - 0.5 × 0.332 = **-0.022**
- **Amber India**: 0.5 × 0.152 - 0.5 × 0.163 = **-0.006**
- **La Victoria**: 0.5 × 0.100 - 0.5 × 0.105 = **-0.003**

**La Victoria** has the highest (least negative) MMR score ← **SELECTED**

### Final Results with λ = 0.5

**Selected Documents** (in order):
1. **Paesano Ristorante Italiano** (Italian) - Most relevant
2. **Osteria Coppa** (Italian) - Still relevant, slightly different
3. **Pho Ha Noi** (Vietnamese) - Moderate relevance, good diversity
4. **Maggianos Little Italy** (Italian) - Fills in Italian cluster
5. **La Victoria Taqueria** (Mexican) - Maximum diversity

### Comparison with Different λ Values

**λ = 0.0 (Maximum Diversity)**:
```
MMR = 0 × Sim(D, Q) - 1.0 × max[Sim(D, Selected)]
    = -max[Sim(D, Selected)]
```
Results would prioritize documents furthest from selected ones:
1. Paesano (first pick, highest relevance)
2. La Victoria (furthest from Paesano)
3. Amber India (furthest from {Paesano, La Victoria})
4. Sushi (adds more diversity)
5. Pho Ha Noi (completes diverse set)

**λ = 1.0 (Maximum Relevance, Standard KNN)**:
```
MMR = 1.0 × Sim(D, Q) - 0 × max[Sim(D, Selected)]
    = Sim(D, Q)
```
Results are pure KNN ranking:
1. Paesano (Italian)
2. Maggianos (Italian)
3. Osteria (Italian)
4. Pho Ha Noi (Vietnamese)
5. Sushi (Japanese)
