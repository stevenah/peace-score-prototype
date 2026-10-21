"use client";

import { useState } from "react";
import { fmtPercent, rowShares, stationLabel, stationShort } from "@/lib/model-admin";

/**
 * One hue, surface to blue: the share of a true station's clips that landed in
 * each predicted station. The ramp stops at 75% blue so the theme's own text
 * colour keeps at least 4.7:1 on every cell, in both themes.
 */
function heat(share: number): string {
  return `color-mix(in oklab, var(--clinical-blue) ${(share * 75).toFixed(1)}%, var(--card))`;
}

interface ConfusionMatrixProps {
  stations: string[];
  /** Rows are the true station, columns the predicted one. */
  confusion: number[][];
}

export function ConfusionMatrix({ stations, confusion }: ConfusionMatrixProps) {
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);
  const shares = rowShares(confusion);
  const rowTotals = confusion.map((row) => row.reduce((a, b) => a + b, 0));

  let readout: React.ReactNode = "Point at a cell to read it out.";
  if (hover) {
    const { row, col } = hover;
    const count = confusion[row][col];
    readout = (
      <>
        <span className="font-semibold text-foreground">
          {count} of {rowTotals[row]}
        </span>{" "}
        {stationLabel(stations[row])} clips ({fmtPercent(shares[row][col])}){" "}
        {row === col ? "were recognised." : `were taken for ${stationLabel(stations[col])}.`}
      </>
    );
  }

  return (
    <figure className="space-y-2">
      <figcaption>
        <p className="text-sm font-medium">Confusion matrix</p>
        <p className="text-xs text-muted-foreground">
          Rows are the true station, columns the predicted one. Numbers are clips.
        </p>
      </figcaption>

      <div className="overflow-x-auto">
        <table
          className="border-separate border-spacing-0.5 text-xs tabular-nums"
          onMouseLeave={() => setHover(null)}
        >
          <thead>
            <tr>
              <td />
              {stations.map((key, col) => (
                <th
                  key={key}
                  scope="col"
                  title={stationLabel(key)}
                  className="h-6 w-9 text-center font-medium text-muted-foreground"
                >
                  {col + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {stations.map((key, row) => (
              <tr key={key}>
                <th
                  scope="row"
                  title={stationLabel(key)}
                  className="whitespace-nowrap pr-2 text-left font-normal text-muted-foreground"
                >
                  <span className="inline-block w-6 font-medium">{row + 1}</span>
                  {stationShort(key)}
                </th>
                {stations.map((_, col) => {
                  const count = confusion[row][col];
                  const active = hover?.row === row && hover?.col === col;
                  return (
                    <td
                      key={col}
                      onMouseEnter={() => setHover({ row, col })}
                      style={{ backgroundColor: heat(shares[row][col]) }}
                      className={`h-8 w-9 rounded-sm text-center text-foreground ${
                        active ? "outline outline-2 outline-ring" : ""
                      }`}
                    >
                      {count > 0 ? count : <span className="sr-only">0</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-1 text-xs text-muted-foreground">
        <p className="min-h-4">{readout}</p>
        <div className="flex items-center gap-2">
          <span>0%</span>
          <span
            aria-hidden="true"
            className="h-2 w-24 rounded-sm"
            style={{ backgroundImage: `linear-gradient(to right, ${heat(0)}, ${heat(1)})` }}
          />
          <span>100% of the row</span>
        </div>
      </div>
    </figure>
  );
}
