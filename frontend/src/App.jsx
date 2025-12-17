// frontend/src/App.jsx
import React, { useMemo, useRef, useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  BarChart,
  Bar,
} from "recharts";
import MetaPanel from "./MetaPanel";

const API = "http://127.0.0.1:5000";

function Card({ title, right, children }) {
  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">{title}</div>
        {right ? <div className="card-right">{right}</div> : null}
      </div>
      {children}
    </div>
  );
}

function UploadBox({ file, setFile, disabled }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);

  function onDrop(e) {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }

  return (
    <div
      className={`drop ${drag ? "drag" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
    >
      <div className="drop-title">Drop .wav here</div>
      <div className="drop-sub">or click Select to choose a file</div>

      <button
        className="btn btn-accent"
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
      >
        Select File
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="audio/wav,.wav"
        className="hidden-input"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />

      {file ? <div className="file-pill">{file.name}</div> : null}
    </div>
  );
}

function SimpleChart({ data, type = "line" }) {
  if (!data || !data.length) {
    return <div className="chart-empty">No data to display yet.</div>;
  }
  const rows = data.map((y, i) => ({ i, y }));

  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height="100%">
        {type === "bar" ? (
          <BarChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
            <XAxis dataKey="i" hide />
            <YAxis />
            <Tooltip />
            <Bar dataKey="y" />
          </BarChart>
        ) : (
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
            <XAxis dataKey="i" hide />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="y" dot={false} />
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

function Table({ columns, rows }) {
  return (
    <div className="table-wrap">
      <table className="tbl">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, idx) => (
            <tr key={idx}>
              {r.map((cell, i) => (
                <td key={i}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState(null);
  const [series, setSeries] = useState({ f0: [], rms: [], zcr: [], mfcc: [] });
  const [featNames, setFeatNames] = useState([]);
  const [featVector, setFeatVector] = useState([]);
  const [showTable, setShowTable] = useState(false);
  const [meta, setMeta] = useState(null);
  const [modelLabel, setModelLabel] = useState(""); // if you want to edit display

  useEffect(() => {
    let mounted = true;
    fetch(`${API}/meta`)
      .then((r) => r.json())
      .then((j) => {
        if (!mounted) return;
        setMeta(j);
        setModelLabel((j && j.model_label) || "RF");
      })
      .catch(() => {
        if (mounted) setMeta(null);
      });
    return () => (mounted = false);
  }, []);

  const verdict = useMemo(() => {
    if (!result) return { text: "No result yet.", tone: "badge" };
    const prob = result?.probability ?? null;
    const used_threshold = result?.used_threshold ?? 0.5;
    const risky = result?.prediction === 1 || (prob !== null && prob >= used_threshold);
    return {
      text: risky ? "Parkinson’s risk" : "Healthy",
      tone: risky ? "badge badge-danger" : "badge badge-okay",
    };
  }, [result]);

  async function analyze() {
    if (!file || busy) return;
    setBusy(true);
    setErr("");
    setResult(null);
    setSeries({ f0: [], rms: [], zcr: [], mfcc: [] });
    setFeatNames([]);
    setFeatVector([]);

    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API}/predict`, { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j?.error || "Prediction failed");
      setResult(j);

      const fd2 = new FormData();
      fd2.append("file", file);
      const r2 = await fetch(`${API}/features`, { method: "POST", body: fd2 });
      const j2 = await r2.json();
      if (!r2.ok) throw new Error(j2?.error || "Feature extraction failed");
      setSeries({
        f0: j2.f0 || [],
        rms: j2.rms || [],
        zcr: j2.zcr || [],
        mfcc: j2.mfcc_means || [],
      });
      setFeatNames(j2.feature_names || []);
      setFeatVector(j2.features_vector || []);
    } catch (e) {
      setErr(e.message || "Failed to fetch");
    } finally {
      setBusy(false);
    }
  }

  const tableRows = useMemo(() => {
    if (!featNames.length || !featVector.length) return [];
    return featNames.map((name, i) => [name, Number(featVector[i]).toFixed(6)]);
  }, [featNames, featVector]);

  return (
    <div className="wrap">
      <header className="top">
        <div className="dot" />
        <h1>Voice Parkinson’s Screening</h1>
      </header>

      <div style={{ marginBottom: 10 }}>
        <label style={{ marginRight: 8 }}>Display model label:</label>
        <input
          value={modelLabel}
          onChange={(e) => setModelLabel(e.target.value)}
          placeholder="e.g. RF"
          style={{ padding: "6px 10px", borderRadius: 6 }}
        />
      </div>

      <MetaPanel apiBase={API} modelLabel={modelLabel || undefined} />

      <p className="sub">
        Upload a .wav clip — we’ll extract features (librosa) and classify with
        your trained model.
      </p>

      <div className="grid">
        <div className="left">
          <Card title="Upload">
            <UploadBox file={file} setFile={setFile} />
            <div className="row" style={{ gap: 12, alignItems: "center", marginTop: 12 }}>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button
                  className="btn btn-accent"
                  disabled={!file || busy}
                  onClick={analyze}
                >
                  {busy ? "Analyzing..." : "Analyze"}
                </button>

                <button
                  className="btn btn-outline"
                  disabled={!featVector.length}
                  onClick={() => setShowTable(true)}
                >
                  Feature set
                </button>
              </div>
            </div>

            {err ? <div className="alert">{err}</div> : null}

            <div className="mini-grid">
              <Card title="Prediction Result">
                <div className="pred">
                  <div className="pred-big">{result?.prediction ?? "—"}</div>
                  <div className="pred-sub">
                    Probability:&nbsp;
                    {result?.probability == null
                      ? "—"
                      : `${(result.probability * 100).toFixed(1)}%`}
                  </div>
                  <div className={verdict.tone} style={{ marginTop: 10 }}>
                    {verdict.text}
                  </div>
                  <div className="note">
                    Using threshold {result?.used_threshold?.toFixed(2) ?? "0.50"}
                  </div>
                </div>
              </Card>

              <Card title="Raw JSON">
                <pre className="json">{JSON.stringify(result || {}, null, 2)}</pre>
              </Card>
            </div>
          </Card>
        </div>

        <div className="right">
          <Card
            title="Feature Graphs"
            right={!series.f0.length ? "No data yet" : "From your uploaded audio"}
          >
            <div className="charts">
              <Card title="F0 (Pitch) over Time">
                <SimpleChart data={series.f0} type="line" />
              </Card>
              <Card title="RMS Energy over Time">
                <SimpleChart data={series.rms} type="line" />
              </Card>
              <Card title="Zero Crossing Rate over Time">
                <SimpleChart data={series.zcr} type="line" />
              </Card>
              <Card title="MFCC Means (13)">
                <SimpleChart data={series.mfcc} type="bar" />
              </Card>
            </div>
          </Card>
        </div>
      </div>

      {showTable && (
        <div className="modal">
          <div className="modal-card">
            <div className="modal-head">
              <div className="card-title">Extracted Feature Set</div>
              <button className="btn btn-outline" onClick={() => setShowTable(false)}>
                Close
              </button>
            </div>
            {!featNames.length ? (
              <div className="muted">No features loaded yet.</div>
            ) : (
              <Table columns={["Feature", "Value"]} rows={tableRows} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}