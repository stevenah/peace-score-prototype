import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { redirect } from "next/navigation";
import Link from "next/link";
import { Upload, Clock, Activity, FileVideo } from "lucide-react";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { PEACE_SCORE_LABELS, PEACE_SCORE_COLORS } from "@/lib/constants";
import type { PeaceScore } from "@/lib/types";

export default async function DashboardPage() {
  const session = await auth();
  if (!session?.user) redirect("/login");

  const analyses = await prisma.analysisSession.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
    take: 20,
  });

  const completedAnalyses = analyses.filter((a) => a.status === "completed");
  const avgScore =
    completedAnalyses.length > 0
      ? completedAnalyses.reduce((sum, a) => sum + (a.overallScore ?? 0), 0) /
        completedAnalyses.length
      : null;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
          Welcome back, {session.user.name || session.user.email}
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          Your analysis overview and history
        </p>
      </div>

      <div className="flex gap-3">
        <Link href="/analyze">
          <Button>
            <Upload className="h-4 w-4" /> New Analysis
          </Button>
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card>
          <div className="flex items-center gap-3 p-1">
            <div className="rounded-lg bg-blue-50 p-2 dark:bg-blue-950">
              <FileVideo className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {analyses.length}
              </p>
              <p className="text-xs text-neutral-500">Total Analyses</p>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex items-center gap-3 p-1">
            <div className="rounded-lg bg-green-50 p-2 dark:bg-green-950">
              <Clock className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {completedAnalyses.length}
              </p>
              <p className="text-xs text-neutral-500">Completed</p>
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex items-center gap-3 p-1">
            <div className="rounded-lg bg-orange-50 p-2 dark:bg-orange-950">
              <Activity className="h-6 w-6 text-orange-600 dark:text-orange-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                {avgScore !== null ? avgScore.toFixed(1) : "---"}
              </p>
              <p className="text-xs text-neutral-500">Avg PEACE Score</p>
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Analyses</CardTitle>
        </CardHeader>
        <div className="px-6 pb-6">
          {analyses.length === 0 ? (
            <p className="py-8 text-center text-sm text-neutral-400">
              No analyses yet. Upload a video to get started.
            </p>
          ) : (
            <div className="space-y-2">
              {analyses.map((analysis) => (
                <Link
                  key={analysis.id}
                  href={`/results/${analysis.analysisId}`}
                  className="flex items-center justify-between rounded-lg border border-neutral-100 p-4 transition-colors hover:bg-neutral-50 dark:border-neutral-800 dark:hover:bg-neutral-800/50"
                >
                  <div>
                    <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {analysis.filename}
                    </p>
                    <p className="text-xs text-neutral-500">
                      {new Date(analysis.createdAt).toLocaleDateString()} at{" "}
                      {new Date(analysis.createdAt).toLocaleTimeString()}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    {analysis.status === "completed" &&
                    analysis.overallScore !== null ? (
                      <span
                        className="text-sm font-bold"
                        style={{
                          color:
                            PEACE_SCORE_COLORS[
                              analysis.overallScore as PeaceScore
                            ],
                        }}
                      >
                        {analysis.overallScore}/3{" "}
                        {
                          PEACE_SCORE_LABELS[
                            analysis.overallScore as PeaceScore
                          ]
                        }
                      </span>
                    ) : (
                      <span className="rounded-full bg-neutral-100 px-2.5 py-0.5 text-xs font-medium text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
                        {analysis.status}
                      </span>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
