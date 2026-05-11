<script lang="ts">
    import type { Snippet } from "svelte";
    import type { HTMLAttributes } from 'svelte/elements';
	let { children, ...rest } : { children: Snippet, disabled?: boolean } & HTMLAttributes<HTMLButtonElement> = $props();
</script>

<button {...rest}>
    {@render children()}
</button>

<style>

    button {
		height: 36px;
		padding: 0 16px;
		border-radius: var(--radius);
		border: 0.5px solid var(--border2);
		color: var(--text);
		font-size: 13px;
		cursor: pointer;
		display: flex;
		align-items: center;
		gap: 6px;
		transition:
			background 0.15s,
			opacity 0.15s;
		background: var(--accent);
    }

    button:hover {
		background: var(--accent2);
    }

    button:active {
		transform: scale(0.98);
    }

    button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
		transform: none;

    }
</style>