<script lang="ts">
	import type { EmbeddingClassifierResults } from '$lib/server';
	import EmbeddingView from './embedding-view.svelte';
	import SimpleDecisionView from './simple-decision-view.svelte';

	const { results, title }: { title: string; results: EmbeddingClassifierResults } = $props();
</script>

<div>
	<div class="embedding-title">
		{title} Embedding ({results.embedding.length}-D)
	</div>
	<EmbeddingView vector={results.embedding} />
	<div class="inference-time">
		Embedding took {(results.embedding_calculation_time * 1000).toFixed(1)} milliseconds.
	</div>

	<div class="decisions-container">
		<!-- {#each vector as value}
			<div class="item" style={`background-color: ${computeCellColor(value)};`}>
				{value.toFixed(2)}
			</div>
		{/each} -->
		<SimpleDecisionView decision={results.svm} title="SVM" />
		<SimpleDecisionView decision={results.naive_bayes} title="Naive Bayes" />
		<!-- <SimpleDecisionView decision={results.nearest_neighbors_cosine} title="K-NN (Cosine)" />
		<SimpleDecisionView decision={results.nearest_neighbors_euclidean} title="K-NN (Euclidean)" />
		<SimpleDecisionView decision={results.nearest_neighbors_minkowski} title="K-NN (Minkowski)" /> -->
		<SimpleDecisionView decision={results.logistic_regression} title="LogReg" />
		<SimpleDecisionView decision={results.neural_network} title="NN" />
	</div>
</div>

<style>
	.decisions-container {
		display: flex;
		flex-direction: row;
		font-family: var(--font-mono);
		font-size: 7px;
		max-width: 100%;
		flex-wrap: wrap;
		gap: 8px;

		margin-top: 12px;
	}

	.embedding-title {
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--text3);
		margin-bottom: 6px;
	}

	.inference-time {
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text3);
		margin-top: 8px;
		padding-left: 8px;
	}
</style>
