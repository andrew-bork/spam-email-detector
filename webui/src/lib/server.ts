

// class SimpleDecision(BaseModel):
//     calculation_time: float
//     decision: bool

// class NearbyNeighbor(BaseModel):
//     text: str
//     distance: float
//     is_spam: bool

// class NearestNeighborsDecision(SimpleDecision):
//     nearest_neighbors: list[NearbyNeighbor]

// class EmbeddingClassifierResults(BaseModel):
//     embedding: list[float]
//     embedding_calculation_time: float

//     svm: SimpleDecision
//     # nearest_neighbors_euclidean: SimpleDecision
//     # nearest_neighbors_minkowski: SimpleDecision
//     # nearest_neighbors_cosine: SimpleDecision
//     nearest_neighbors_euclidean: NearestNeighborsDecision
//     nearest_neighbors_minkowski: NearestNeighborsDecision
//     nearest_neighbors_cosine: NearestNeighborsDecision
//     naive_bayes: SimpleDecision
//     logistic_regression: SimpleDecision
//     neural_network: SimpleDecision

// class NonEmbeddingClassifiersResults(BaseModel):
//     bert_classifier: SimpleDecision
    
// class InferencerResults(BaseModel):
//     sentence_transformer: EmbeddingClassifierResults
//     word_t_vec_trained: EmbeddingClassifierResults
//     word_t_vec_pretrained: EmbeddingClassifierResults
//     non_embedding: NonEmbeddingClassifiersResults
export type SimpleDecision = {
    calculation_time: number;
    decision: boolean;
};
export type NearbyNeighbor = {
    text: string;
    distance: number;
    is_spam: boolean;
}
export type NearestNeighborsDecision = SimpleDecision & {
    nearest_neighbors: NearbyNeighbor[]
};
export type EmbeddingClassifierResults = {
    embedding: number[];
    embedding_calculation_time: number;
    svm: SimpleDecision;
    naive_bayes: SimpleDecision;
    logistic_regression: SimpleDecision;
    neural_network: SimpleDecision;
    nearest_neighbors_euclidean: NearestNeighborsDecision;
    nearest_neighbors_minkowski: NearestNeighborsDecision;
    nearest_neighbors_cosine: NearestNeighborsDecision;
}

export type NonEmbeddingClassifiersResults = {
    bert_classifier: SimpleDecision
}
export type InferenceResultType = {
    sentence_transformer: EmbeddingClassifierResults;
    word_t_vec_trained: EmbeddingClassifierResults;
    word_t_vec_pretrained: EmbeddingClassifierResults;
    non_embedding: NonEmbeddingClassifiersResults;
}

export class WebuiClient {
    webuiServerHost: string;

    constructor(webuiServerHost: string = "http://localhost:1111") {
        this.webuiServerHost = webuiServerHost;
    }

    _buildURL(endpoint: string) {
        return `${this.webuiServerHost}${endpoint}`;
    }

    async _post<ResultType>(endpoint: string, body: unknown) : Promise<ResultType> {
        const result = await fetch(
            this._buildURL(endpoint), 
            {
                method: "POST",
                headers: {
                    "content-type": "application/json"
                },
                body: JSON.stringify(body)
            });
        
        if(result.status !== 200) throw new Error(result.statusText);
        
        return result.json() as ResultType;
    }

    async runAllInferences(input: string) {
        return await this._post<InferenceResultType>("/api/infer/all", {
            input
        });
    }
}