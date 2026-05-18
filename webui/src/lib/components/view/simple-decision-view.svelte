<script lang="ts">
	import type { SimpleDecision } from '$lib/server';

	const { title, decision }: { title: string; decision: SimpleDecision } = $props();
</script>

<div class="container">
	<div class="title">{title}</div>
	{#if decision.decision}
		<div class="decision">spam</div>
	{:else}
		<div class="decision">not spam</div>
	{/if}
	<div class="inference-time">
		{(decision.calculation_time * 1000).toFixed(1)} ms
	</div>
</div>

<style>
	.title {
		font-size: 12px;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--text2);
	}
	.decision {
		font-family: var(--font-mono);
		font-size: 10px;
	}
	.container {
		display: flex;
		flex-direction: column;
		font-family: var(--font-mono);
		font-size: 10px;
		flex-wrap: wrap;
		width: 20%;

		background: var(--bg2);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 8px;
	}

	.inference-time {
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text3);
		/* margin-top: 8px; */
		/* padding-left: 8px; */
	}

	/* .item {
		display: flex;
		align-items: center;
		justify-content: center;
		min-width: 24px;
		min-height: 24px;
	} */
</style>
