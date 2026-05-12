

export type InferenceResultType = {
    sentence_transformer_encode_time: number;
    sentence_transformer_embedding: number[];
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