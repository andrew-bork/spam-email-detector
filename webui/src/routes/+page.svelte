<script lang="ts">
	import CardTitle from '$lib/components/card/card-title.svelte';
	import Card from '$lib/components/card/card.svelte';
	import SpecialButton from '$lib/components/button/special-button.svelte';
	import TextArea from '$lib/components/input/text-area.svelte';
	import SecondaryButton from '$lib/components/button/secondary-button.svelte';
	import { WebuiClient, type InferenceResultType } from '$lib/server';
	import EmbeddingView from '$lib/components/view/embedding-view.svelte';

	let inferenceResult = $state({
		sentence_transformer_embedding: [],
		sentence_transformer_encode_time: 0
	} as InferenceResultType);
	let inferenceInput = $state('');
	const canRunInference = $derived(0 < inferenceInput.trim().length);
</script>

<div class="shell">
	<main class="page">
		<div class="page-title">Run Inference</div>
		<div class="page-sub">Classify text using all models.</div>

		<Card>
			<div class="page-title">Run Inference</div>
			<div class="page-sub">Classify text using all models.</div>
			<CardTitle>Input</CardTitle>
			<CardTitle style="margin-bottom:8px; font-size:10px;">Try an example</CardTitle>

			<div style="position: relative;">
				<TextArea
					placeholder="Paste or type a message to classify…"
					rows={10}
					oninput={(e) => {
						inferenceInput = e.currentTarget.value;
					}}
					value={inferenceInput}
				></TextArea>
				<span class="token-count">0</span>
			</div>
			<!-- <div id="token-hint" class="token-preview" style="margin-top:6px;"></div> -->

			<div class="btn-row">
				<SpecialButton
					disabled={!canRunInference}
					onclick={async () => {
						const client = new WebuiClient('http://localhost:8000');
						const result = await client.runAllInferences(inferenceInput);
						inferenceResult = result;
					}}
				>
					<svg width="13" height="13" viewBox="0 0 16 16" fill="white"
						><path d="M2 4h12M2 8h12M2 12h12" stroke="white" stroke-width="1.5" fill="none" /></svg
					>
					Run All Classifiers
				</SpecialButton>

				<SecondaryButton
					class="btn btn-secondary"
					onclick={() => {
						inferenceInput = '';
					}}
				>
					Clear
				</SecondaryButton>
			</div>
		</Card>

		<Card>
			<CardTitle>Inference Results</CardTitle>
			{#if inferenceResult.sentence_transformer_embedding.length == 0}
				<div>No results yet.</div>
			{:else}
				<div class="embedding-title">
					Sentence Transformer Embedding ({inferenceResult.sentence_transformer_embedding.length}-D)
				</div>
				<EmbeddingView vector={inferenceResult.sentence_transformer_embedding} />
				<div class="inference-time">
					Embedding took {(inferenceResult.sentence_transformer_encode_time * 1000).toFixed(1)} milliseconds.
				</div>
			{/if}
		</Card>
	</main>
</div>

<style>
	.page-title {
		font-size: 20px;
		font-weight: 700;
		letter-spacing: -0.02em;
		margin-bottom: 4px;
		padding-left: 8px;
	}
	.page-sub {
		font-size: 13px;
		color: var(--text2);
		margin-bottom: 28px;
		padding-left: 16px;
	}
	.page {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: flex-start;
	}

	.page > :global(*) {
		width: calc(100% - 500px);
	}

	/* Layout */
	.shell {
		height: 100%;
		display: grid;
		/* grid-template-columns: 280px auto; */
		grid-template-columns: 100%;
		/* min-height: 100vh;  */
		width: 100%;
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

	.token-count {
		position: absolute;
		bottom: 10px;
		right: 12px;
		font-size: 10px;
		font-family: var(--font-mono);
		color: var(--text3);
		pointer-events: none;
	}

	.btn-row {
		display: flex;
		gap: 10px;
		align-items: center;
		margin-top: 16px;
		flex-wrap: wrap;
	}

	main {
		padding: 16px;
		max-width: 100%;
	}

	/* Sidebar */
	aside {
		border-right: 0.5px solid var(--border);
		overflow-y: auto;
		padding: 16px;
		display: flex;
		flex-direction: column;
		gap: 20px;
		background: var(--bg2);
	}
</style>
