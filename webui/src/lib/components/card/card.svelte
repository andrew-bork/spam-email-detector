<script lang="ts">
    import type { Snippet } from "svelte";
    import type { HTMLAttributes } from 'svelte/elements';
	let { children, ...rest } : { children: Snippet } & HTMLAttributes<HTMLDivElement> = $props();
</script>

<div {...rest}>
    {@render children()}
</div>

<style>
    div {
		background: var(--bg2);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 20px 22px;
		margin-bottom: 16px;
    }
</style>