import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { redirect } from "next/navigation";
import { AnalysisRow } from "./AnalysisRow";

export default async function DashboardPage() {
  const session = await auth();
  if (!session?.user) redirect("/login");

  const analyses = await prisma.analysisSession.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
    take: 20,
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
          Welcome back, {session.user.name || session.user.email}
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          Your analysis overview and history
        </p>
      </div>

      <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
        Recent Analyses
      </h2>

      {analyses.length === 0 ? (
        <p className="py-8 text-center text-sm text-neutral-400">
          No analyses yet. Upload a video to get started.
        </p>
      ) : (
        <div className="space-y-2">
          {analyses.map((analysis) => (
            <AnalysisRow key={analysis.id} analysis={analysis} />
          ))}
        </div>
      )}
    </div>
  );
}
