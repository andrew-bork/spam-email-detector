<script>
	import CardTitle from '$lib/components/card/card-title.svelte';
	import Card from '$lib/components/card/card.svelte';
	import SpecialButton from '$lib/components/button/special-button.svelte';
	import TextArea from '$lib/components/input/text-area.svelte';



	let inferenceInput = $state("");
	const canRunInference = $derived(0 < inferenceInput.trim().length);

</script>

<div class="shell">
	<aside>
		<h1>Inference</h1>
	</aside>
	<main class="page">
		<div class="page-title">Run Inference</div>
		<div class="page-sub">Classify text using all models.</div>

		<Card>
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
				<span
					id="infer-charcount"
					style="position:absolute; bottom:10px; right:12px; font-size:10px; font-family:var(--font-mono); color:var(--text3); pointer-events:none;"
					>0</span
				>
			</div>
			<div id="token-hint" class="token-preview" style="margin-top:6px;"></div>

			<div class="btn-row">
				<SpecialButton 
					disabled={!canRunInference}
					onclick={() => {
						alert(inferenceInput);
					}}
				>
					<svg width="13" height="13" viewBox="0 0 16 16" fill="white"
						><path d="M2 4h12M2 8h12M2 12h12" stroke="white" stroke-width="1.5" fill="none" /></svg
					>
					Run All Classifiers
				</SpecialButton>

				<!-- <PrimaryButton disabled>
					<svg width="13" height="13" viewBox="0 0 16 16" fill="white"
						><path d="M2 4h12M2 8h12M2 12h12" stroke="white" stroke-width="1.5" fill="none" /></svg
					>
					Run All Classifiers
				</PrimaryButton> -->

				<button 
					class="btn btn-secondary"
					onclick={() => {
						inferenceInput = "";
					}}
				>Clear</button>
			</div>
		</Card>

		<!-- Results area -->
		<div id="infer-result-area"></div>
	</main>
</div>

<style>
	.page-title {
		font-size: 20px;
		font-weight: 700;
		letter-spacing: -0.02em;
		margin-bottom: 4px;
	}
	.page-sub {
		font-size: 13px;
		color: var(--text2);
		margin-bottom: 28px;
	}
	
	/* Layout */
	.shell {
		height: 100%;
		display: grid;
		grid-template-columns: 280px 1fr;
		/* min-height: 100vh;  */
	}

	button {
		height: 36px;
		padding: 0 16px;
		border-radius: var(--radius);
		border: 0.5px solid var(--border2);
		background: transparent;
		color: var(--text);
		font-size: 13px;
		cursor: pointer;
		display: flex;
		align-items: center;
		gap: 6px;
		transition:
			background 0.15s,
			opacity 0.15s;
	}
	button:hover {
		background: var(--bg3);
	}
	button:active {
		transform: scale(0.98);
	}
	button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	/* ── Buttons ── */
	.btn {
		display: inline-flex;
		align-items: center;
		gap: 7px;
		padding: 9px 18px;
		border-radius: var(--radius);
		border: none;
		font-family: var(--font-sans);
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.15s;
	}
	.btn-secondary {
		background: var(--bg3);
		color: var(--text);
		border: 1px solid var(--border);
	}
	.btn-secondary:hover {
		background: var(--bg4);
		border-color: var(--border-hi);
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
