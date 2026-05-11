<script lang="ts">
    import type { HTMLAttributes } from 'svelte/elements';
	let props : { rows: number } & HTMLAttributes<HTMLTextAreaElement> = $props();
</script>

<textarea
    {...props}
></textarea>

<style>
	textarea {
		background: var(--bg3);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		color: var(--text);
		font-family: var(--font-mono);
		font-size: 13px;
		padding: 10px 12px;
		outline: none;
		width: 100%;
		resize: vertical;
		min-height: 100px;
		transition: border-color 0.15s;
		line-height: 1.6;
	}
	textarea:focus {
		border-color: var(--accent-border);
	}
</style>