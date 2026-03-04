import Link from "next/link";
import { Upload, Radio, ArrowRight } from "lucide-react";
import { Card } from "@/components/ui/Card";

const options = [
  {
    href: "/analyze/upload",
    icon: Upload,
    title: "Upload Video",
    description: "Upload one or more videos for batch processing and analysis",
  },
  {
    href: "/analyze/live",
    icon: Radio,
    title: "Live Analysis",
    description: "Perform real-time frame-by-frame analysis on a video",
  },
];

export default function AnalyzePage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Video Analysis</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Choose an analysis method to get started
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        {options.map((opt) => (
          <Link key={opt.href} href={opt.href} className="group">
            <Card className="flex h-full flex-col items-center justify-center gap-4 px-8 py-12 text-center transition-colors hover:border-primary/40 hover:bg-accent/50">
              <div className="rounded-full bg-primary/10 p-4">
                <opt.icon className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-foreground">
                  {opt.title}
                </h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {opt.description}
                </p>
              </div>
              <span className="inline-flex items-center gap-1 text-sm font-medium text-primary opacity-0 transition-opacity group-hover:opacity-100">
                Get started <ArrowRight className="h-4 w-4" />
              </span>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
