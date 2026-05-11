



class WebuiClient {
    webuiServerHost: string;

    constructor(webuiServerHost: string = "localhost:1111") {
        this.webuiServerHost = webuiServerHost;
    }

    _buildURL(endpoint: string) {
        return `htts://${}${endpoint}`;
    }

    async _post<RequestType, ResultType>(endpoint: string, body: RequestType) : Promise<ResultType> {
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
        return JSON.stringify(result.body) as ResultType;
    }

    async runAllInferences(input: string) {
        return await this._post("/api/infer/all", {
            input
        });
    }
}