// frontend/src/MetaPanel.jsx
import "./MetaPanel.css";
import React, { useEffect, useState } from "react";

export default function MetaPanel({ apiBase, modelLabel }) {
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    let mounted = true;
    fetch(`${apiBase}/meta`)
      .then((r) => r.json())
      .then((j) => {
        if (mounted) setMeta(j);
      })
      .catch(() => {
        if (mounted) setMeta(null);
      });
    return () => {
      mounted = false;
    };
  }, [apiBase]);

  const model =
    (meta && (meta.model_label || meta.chosen_model || "RF")) ||
    (modelLabel || "RF");

  const accuracy =
    meta && meta.accuracy != null
      ? `${(meta.accuracy * 100).toFixed(1)}%`
      : "—";

  const features = meta && meta.n_features != null ? meta.n_features : "—";

  const dist = meta && meta.class_distribution ? meta.class_distribution : {};

  const hc_count =
    (dist["0"] !== undefined ? dist["0"] : dist[0]) || 0;
  const pd_count =
    (dist["1"] !== undefined ? dist["1"] : dist[1]) || 0;

  return (
    <div className="meta-wrap">
      <div className="meta-row">
        <div className="meta-card">
          <div className="meta-small">Model</div>
          <div className="meta-big">{model}</div>
        </div>

        <div className="meta-card">
          <div className="meta-small">Accuracy</div>
          <div className="meta-big">{accuracy}</div>
          <div className="meta-note">on validation/test split</div>
        </div>

        <div className="meta-card">
          <div className="meta-small">Features</div>
          <div className="meta-big">{features}</div>
        </div>

        <div className="meta-card">
          <div className="meta-small">HC (healthy)</div>
          <div className="meta-big">{hc_count}</div>
        </div>

        <div className="meta-card">
          <div className="meta-small">PD (patients)</div>
          <div className="meta-big">{pd_count}</div>
        </div>
      </div>
    </div>
  );
}