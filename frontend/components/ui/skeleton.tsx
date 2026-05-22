import { cn } from "@/lib/utils";

/**
 * Skeleton placeholder for loading states. Standard shadcn shape:
 * `animate-pulse` + theme-aware muted background. Reads from CSS variables
 * so light/dark mode just works.
 */
export function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-[4px] bg-[var(--surface-hover)]",
        className,
      )}
      {...props}
    />
  );
}
