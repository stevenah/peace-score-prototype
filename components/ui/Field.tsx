import { cn } from "@/lib/utils";
import { Label } from "./Label";

interface FieldProps {
  label: React.ReactNode;
  htmlFor?: string;
  /** Marks the label with an asterisk. The control still needs `required`. */
  required?: boolean;
  /** Small help text under the control. */
  hint?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}

/** Label, control and hint with one spacing rhythm for every form. */
export function Field({ label, htmlFor, required, hint, className, children }: FieldProps) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <Label htmlFor={htmlFor}>
        {label}
        {required && (
          <span aria-hidden="true" className="-ml-1 text-destructive">
            *
          </span>
        )}
      </Label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}
