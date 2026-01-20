import { useEffect, useMemo, useState } from "react";

type NextRace = {
  season: number;
  round: number;
  raceName: string;
  date?: string;
  time?: string;
  circuit?: string;
  location?: { locality?: string; country?: string; lat?: string; long?: string };
};

type GridRow = { driver: string; constructor: string; grid: number };
type RosterRow = { driver: string; constructor: string; standing_position?: number };

type PredictRow = { driver: string; constructor: string; grid: number; win_probability: number };

const API = import.meta.env.VITE_API || "http://127.0.0.1:8000";

function fmtRace(r?: NextRace | null) {
  if (!r) return "";
  const loc = r.location?.locality && r.location?.country ? `${r.location.locality}, ${r.location.country}` : "";
  const when = r.date ? `${r.date}${r.time ? ` ${r.time}` : ""}` : "";
  return `${r.raceName}${loc ? ` • ${loc}` : ""}${when ? ` • ${when}` : ""}`;
}

export default function App() {
  const [loading, setLoading] = useState(true);
  const [race, setRace] = useState<NextRace | null>(null);

  const [mode, setMode] = useState<"AUTO" | "MANUAL_GRID">("AUTO"); // AUTO when qualifying grid exists
  const [gridRows, setGridRows] = useState<GridRow[]>([]);
  const [manualGrid, setManualGrid] = useState<Record<string, string>>({}); // driver -> grid string input

  const [predLoading, setPredLoading] = useState(false);
  const [predictions, setPredictions] = useState<PredictRow[]>([]);
  const [error, setError] = useState<string>("");

  // Load race + grid/roster
  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        setError("");

        // Next race
        const rRace = await fetch(`${API}/api/next-race`);
        const raceJson = await rRace.json();
        if (raceJson?.error) throw new Error(raceJson.error);
        setRace(raceJson);

        // Next grid (if qualifying available)
        const rGrid = await fetch(`${API}/api/next-grid`);
        const gridJson = await rGrid.json();
        const g: GridRow[] = gridJson?.grid || [];

        if (g.length > 0) {
          setMode("AUTO");
          setGridRows(g);
          setManualGrid({});
          setPredictions([]);
        } else {
          // Fall back to roster (drivers + teams), user fills grid
          setMode("MANUAL_GRID");
          const rRoster = await fetch(`${API}/api/next-roster`);
          const rosterJson = await rRoster.json();
          if (rosterJson?.error) throw new Error(rosterJson.error);

          const roster: RosterRow[] = rosterJson?.drivers || [];
          const rows: GridRow[] = roster.map((d) => ({
            driver: d.driver,
            constructor: d.constructor,
            grid: 0,
          }));
          setGridRows(rows);

          // Seed empty inputs
          const seed: Record<string, string> = {};
          for (const row of rows) seed[row.driver] = "";
          setManualGrid(seed);
          setPredictions([]);
        }
      } catch (e: any) {
        setError(e?.message || "Failed to load");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const title = useMemo(() => fmtRace(race), [race]);

  function updateGrid(driver: string, value: string) {
    // Keep as string in state so user can type freely
    setManualGrid((prev) => ({ ...prev, [driver]: value }));
  }

  function validateManualGrid(): { ok: boolean; message?: string; payload?: any } {
    const entries = gridRows.map((row) => {
      const v = (manualGrid[row.driver] ?? "").trim();
      const n = Number(v);
      return { driver: row.driver, grid: n, raw: v };
    });

    // basic checks
    for (const e of entries) {
      if (!e.raw) return { ok: false, message: `Missing grid for ${e.driver}` };
      if (!Number.isFinite(e.grid) || e.grid <= 0) return { ok: false, message: `Invalid grid for ${e.driver}` };
      if (!Number.isInteger(e.grid)) return { ok: false, message: `Grid must be an integer for ${e.driver}` };
    }

    // (Optional) ensure uniqueness 1..20 (light validation)
    const set = new Set(entries.map((e) => e.grid));
    if (set.size !== entries.length) return { ok: false, message: "Grid values must be unique" };

    return { ok: true, payload: { grid: entries.map((e) => ({ driver: e.driver, grid: e.grid })) } };
  }

  async function predictAll() {
    try {
      setPredLoading(true);
      setError("");
      setPredictions([]);

      if (mode === "AUTO") {
        const r = await fetch(`${API}/api/predict-next-race`);
        const j = await r.json();
        if (j?.error) throw new Error(j.error);
        if (j?.message && (!j?.predictions || j.predictions.length === 0)) {
          // qualifying not available, backend tells us to go manual
          throw new Error(j.message);
        }
        setPredictions(j.predictions || []);
      } else {
        const v = validateManualGrid();
        if (!v.ok) throw new Error(v.message);

        const r = await fetch(`${API}/api/predict-next-race-with-grid`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(v.payload),
        });
        const j = await r.json();
        if (j?.error) throw new Error(j.error === "validation_failed" ? JSON.stringify(j.errors) : j.error);
        setPredictions(j.predictions || []);
      }
    } catch (e: any) {
      setError(e?.message || "Prediction failed");
    } finally {
      setPredLoading(false);
    }
  }

  return (
    <div style={{ fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif", background: "linear-gradient(135deg, #3E3B39 0%, #4A4744 100%)", minHeight: "100vh", padding: "40px 20px", width: "100%", boxSizing: "border-box" }}>
      <div style={{ width: "100%", maxWidth: "1400px", margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 40 }}>
          <h1 style={{ margin: 0, fontSize: 48, fontWeight: 800, background: "linear-gradient(135deg, #8B9D83 0%, #A8B89F 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
            F1 Predictor
          </h1>
          <div style={{ fontSize: 18, color: "#cbd5e1", marginTop: 12, fontWeight: 500 }}>{title || "Loading race information..."}</div>
        </div>

        {/* Status & Controls */}
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 24, flexWrap: "wrap" }}>
          <div
            style={{
              fontSize: 13,
              padding: "8px 16px",
              borderRadius: 20,
              background: mode === "AUTO" ? "linear-gradient(135deg, #8B9D83 0%, #7A8C7A 100%)" : "linear-gradient(135deg, #D4C4B0 0%, #C9B8A8 100%)",
              color: "#fff",
              fontWeight: 600,
              boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
            }}
          >
            {mode === "AUTO" ? "Qualifying Data Available" : "Manual Grid Entry"}
          </div>

          <button
            onClick={predictAll}
            disabled={loading || predLoading}
            style={{
              padding: "10px 24px",
              borderRadius: 20,
              border: "none",
              background: loading || predLoading ? "#64748b" : "linear-gradient(135deg, #8B9D83 0%, #A8B89F 100%)",
              color: "#fff",
              cursor: loading || predLoading ? "not-allowed" : "pointer",
              fontWeight: 700,
              fontSize: 15,
              boxShadow: "none",
              transition: "all 0.3s ease",
              opacity: loading || predLoading ? 0.7 : 1,
            }}
          >
            {predLoading ? "Predicting..." : "Predict All"}
          </button>

          <button
            onClick={() => window.location.reload()}
            style={{
              padding: "10px 24px",
              borderRadius: 20,
              border: "2px solid #64748b",
              background: "transparent",
              color: "#e2e8f0",
              cursor: "pointer",
              fontWeight: 700,
              fontSize: 15,
              transition: "all 0.3s ease",
            }}
          >
            Refresh
          </button>
        </div>

        {/* Error Alert */}
        {error ? (
          <div style={{ marginBottom: 24, padding: 16, border: "2px solid #dc2626", background: "rgba(220, 38, 38, 0.1)", borderRadius: 12, backdropFilter: "blur(10px)" }}>
            <div style={{ fontWeight: 700, color: "#fca5a5", marginBottom: 6 }}>Error</div>
            <div style={{ whiteSpace: "pre-wrap", color: "#fecaca", fontSize: 14 }}>{error}</div>
          </div>
        ) : null}

        {/* Main Content */}
        {loading ? (
          <div style={{ textAlign: "center", padding: 60, color: "#cbd5e1" }}>
            <div style={{ fontSize: 20, marginBottom: 12, fontWeight: 600 }}>Loading race data...</div>
          </div>
        ) : (
          <>
            {/* Drivers Table */}
            <div style={{ background: "rgba(62, 59, 57, 0.6)", border: "1px solid rgba(139, 157, 131, 0.3)", borderRadius: 16, overflow: "hidden", marginBottom: 24, backdropFilter: "blur(10px)" }}>
              <div style={{ padding: 20, borderBottom: "1px solid rgba(139, 157, 131, 0.3)", background: "linear-gradient(135deg, rgba(62, 59, 57, 0.8) 0%, rgba(74, 71, 68, 0.8) 100%)" }}>
                <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#f1f5f9" }}>Grid & Predictions</h2>
              </div>

              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ textAlign: "left", background: "rgba(74, 71, 68, 0.5)" }}>
                      <th style={{ padding: "14px 16px", color: "#cbd5e1", fontWeight: 700, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.5px" }}>Driver</th>
                      <th style={{ padding: "14px 16px", color: "#cbd5e1", fontWeight: 700, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.5px" }}>Team</th>
                      <th style={{ padding: "14px 16px", color: "#cbd5e1", fontWeight: 700, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.5px", width: 100 }}>Grid</th>
                      <th style={{ padding: "14px 16px", color: "#cbd5e1", fontWeight: 700, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.5px", textAlign: "right", width: 140 }}>Win Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gridRows.map((row, idx) => {
                      const pred = predictions.find((p) => p.driver === row.driver);
                      const winPct = pred ? pred.win_probability * 100 : null;
                      const isTopThree = predictions.length > 0 && predictions.findIndex(p => p.driver === row.driver) < 3;

                      return (
                        <tr
                          key={row.driver}
                          style={{
                            background: idx % 2 === 0 ? "rgba(15, 23, 42, 0.3)" : "rgba(30, 41, 59, 0.2)",
                            borderBottom: "1px solid rgba(100, 116, 139, 0.1)",
                            transition: "all 0.3s ease",
                            boxShadow: isTopThree ? "inset 0 0 10px rgba(255, 107, 53, 0.1)" : "none",
                          }}
                        >
                          <td style={{ padding: "14px 16px", color: "#f1f5f9", fontWeight: 600, fontSize: 14 }}>
                            {isTopThree && <span style={{ marginRight: 8, color: "#f59e0b", fontWeight: 900 }}>★</span>}
                            {row.driver}
                          </td>
                          <td style={{ padding: "14px 16px", color: "#cbd5e1", fontSize: 14 }}>{row.constructor}</td>
                          <td style={{ padding: "14px 16px", color: "#cbd5e1", fontSize: 14 }}>
                            {mode === "AUTO" ? (
                              <span style={{ fontWeight: 700, color: "#10b981" }}>{row.grid}</span>
                            ) : (
                              <input
                                value={manualGrid[row.driver] ?? ""}
                                onChange={(e) => updateGrid(row.driver, e.target.value)}
                                placeholder="1-20"
                                style={{
                                  width: 80,
                                  padding: "8px 12px",
                                  borderRadius: 8,
                                  border: "2px solid rgba(100, 116, 139, 0.3)",
                                  background: "rgba(30, 41, 59, 0.5)",
                                  color: "#f1f5f9",
                                  fontSize: 14,
                                  fontWeight: 600,
                                  textAlign: "center",
                                }}
                              />
                            )}
                          </td>
                          <td style={{ padding: "14px 16px", textAlign: "right" }}>
                            {winPct !== null ? (
                              <div style={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "flex-end",
                                gap: 8,
                              }}>
                                <div style={{
                                  width: 60,
                                  height: 24,
                                  background: "rgba(100, 116, 139, 0.2)",
                                  borderRadius: 12,
                                  overflow: "hidden",
                                }}>
                                  <div
                                    style={{
                                      width: `${winPct}%`,
                                      height: "100%",
                                      background: `linear-gradient(90deg, ${winPct > 50 ? "#10b981" : "#f59e0b"} 0%, ${winPct > 50 ? "#059669" : "#d97706"} 100%)`,
                                      transition: "width 0.3s ease",
                                    }}
                                  />
                                </div>
                                <span style={{ fontWeight: 700, color: "#f1f5f9", minWidth: 50, textAlign: "right" }}>
                                  {winPct.toFixed(1)}%
                                </span>
                              </div>
                            ) : (
                              <span style={{ color: "#64748b" }}>—</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Info Box */}
            <div style={{ background: "rgba(15, 23, 42, 0.6)", border: "1px solid rgba(100, 116, 139, 0.3)", borderRadius: 16, padding: 20, backdropFilter: "blur(10px)" }}>
              <div style={{ fontWeight: 700, marginBottom: 16, fontSize: 18, color: "#f1f5f9" }}>How Predictions Are Calculated</div>
              
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontWeight: 600, color: "#ff6b35", marginBottom: 12, fontSize: 14 }}>Input Features </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12 }}>
                  <div style={{ padding: 12, background: "rgba(16, 185, 129, 0.1)", borderLeft: "4px solid #10b981", borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, color: "#10b981", marginBottom: 4, fontSize: 13 }}>Grid Position</div>
                    <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>Starting position (1-20) - determines initial momentum and tire strategy exposure</div>
                  </div>
                  <div style={{ padding: 12, background: "rgba(59, 130, 246, 0.1)", borderLeft: "4px solid #3b82f6", borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, color: "#3b82f6", marginBottom: 4, fontSize: 13 }}>Driver Career Points</div>
                    <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>Total championship points accumulated before this race - measures overall driver skill</div>
                  </div>
                  <div style={{ padding: 12, background: "rgba(139, 92, 246, 0.1)", borderLeft: "4px solid #8b5cf6", borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, color: "#8b5cf6", marginBottom: 4, fontSize: 13 }}>5-Race Rolling Average (Points)</div>
                    <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>Recent form - average points scored in last 5 races (excluding current race)</div>
                  </div>
                  <div style={{ padding: 12, background: "rgba(168, 85, 247, 0.1)", borderLeft: "4px solid #a855f7", borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, color: "#a855f7", marginBottom: 4, fontSize: 13 }}>5-Race Finishing Position Average</div>
                    <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>Consistency indicator - average race finishing position over recent races</div>
                  </div>
                  <div style={{ padding: 12, background: "rgba(244, 63, 94, 0.1)", borderLeft: "4px solid #f43f5e", borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, color: "#f43f5e", marginBottom: 4, fontSize: 13 }}>DNF (Did Not Finish) Rate</div>
                    <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>Reliability metric - percentage of races not completed in last 5 races</div>
                  </div>
                  <div style={{ padding: 12, background: "rgba(34, 197, 94, 0.1)", borderLeft: "4px solid #22c55e", borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, color: "#22c55e", marginBottom: 4, fontSize: 13 }}>Team Championship Points</div>
                    <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>Constructor's accumulated points before this race - team resource & performance level</div>
                  </div>
                  <div style={{ padding: 12, background: "rgba(251, 146, 60, 0.1)", borderLeft: "4px solid #fb923c", borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, color: "#fb923c", marginBottom: 4, fontSize: 13 }}>Team 5-Race Average</div>
                    <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>Constructor's recent form - average points per race in last 5 races</div>
                  </div>
                  <div style={{ padding: 12, background: "rgba(59, 130, 246, 0.1)", borderLeft: "4px solid #3b82f6", borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, color: "#3b82f6", marginBottom: 4, fontSize: 13 }}>Season Progress</div>
                    <div style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.5 }}>Normalized round number (early/mid/late season effects on performance)</div>
                  </div>
                </div>
              </div>

              <div style={{ marginBottom: 20 }}>
                <div style={{ fontWeight: 600, color: "#ef4444", marginBottom: 12, fontSize: 14 }}>Important Limitations </div>
                <div style={{ padding: 14, background: "rgba(239, 68, 68, 0.1)", borderLeft: "4px solid #ef4444", borderRadius: 8 }}>
                  <ul style={{ margin: 0, paddingLeft: 18, color: "#fecaca", fontSize: 12, lineHeight: 1.8 }}>
                    <li><strong>Track Characteristics:</strong> No circuit layout, length, corner type, or track history data</li>
                    <li><strong>Weather Conditions:</strong> Rain, temperature, wind, or track surface conditions unknown</li>
                    <li><strong>Team Performance Curve:</strong> No data on car upgrades, performance evolution, or development trajectory</li>
                    <li><strong>Car Specifications:</strong> No aerodynamic changes, setup modifications, or hardware differences</li>
                    <li><strong>Season Form Trends:</strong> Cannot predict momentum shifts or mid-season performance changes</li>
                    <li><strong>Teammate Comparison:</strong> No relative performance data between drivers in the same team</li>
                    <li><strong>Penalty Data:</strong> Grid penalties, technical issues, or incident probabilities not included</li>
                    <li><strong>Historical Head-to-Head:</strong> No track-specific or driver-vs-driver matchup history</li>
                  </ul>
                </div>
              </div>

              <div style={{ padding: 14, background: "rgba(255, 107, 53, 0.1)", borderLeft: "4px solid #ff6b35", borderRadius: 8, fontSize: 12, color: "#cbd5e1", lineHeight: 1.7 }}>
                <strong style={{ color: "#ff6b35", fontSize: 13 }}>Model: XGBoost Classifier</strong> (400 decision trees) trained on historical F1 race data from multiple seasons. The model learns patterns from past race outcomes based on the input features . Predictions are probabilistic estimates based on historical patterns, not guaranteed outcomes. Results are most reliable when used at grid finalization and less reliable as the season progresses without regular retraining.
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

