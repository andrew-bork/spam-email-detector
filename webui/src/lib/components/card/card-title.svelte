<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	let { children, ...rest }: { children: Snippet } & HTMLAttributes<HTMLDivElement> = $props();
</script>

<div {...rest}>
	{@render children()}
</div>

<style>
	div {
		font-size: 12px;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--text2);
		margin-bottom: 14px;
	}
</style>
