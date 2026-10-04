-- AlterTable: ESGE station summary for live analyses. Nullable, no backfill:
-- existing rows and feature-off saves simply have no station data.
ALTER TABLE "AnalysisSession" ADD COLUMN "stationsData" TEXT;
