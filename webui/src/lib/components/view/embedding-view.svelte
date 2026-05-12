<script lang="ts">
	const { vector }: { vector: number[] } = $props();

	function logistic(value: number): number {
		return 1 / (1 + Math.exp(-value));
	}
	function computeCellColor(value: number): string {
		const a = 0.5 * logistic(value * 20);
		return `rgba(0, 255, 0, ${a})`;
	}
</script>

<div class="container">
	{#each vector as value}
		<div class="item" style={`background-color: ${computeCellColor(value)};`}>
			{value.toFixed(2)}
		</div>
	{/each}
</div>

<style>
	.container {
		display: flex;
		flex-direction: row;
		font-family: var(--font-mono);
		font-size: 7px;
		max-width: 100%;
		flex-wrap: wrap;
	}

	.item {
		display: flex;
		align-items: center;
		justify-content: center;
		min-width: 24px;
		min-height: 24px;
	}
</style>
