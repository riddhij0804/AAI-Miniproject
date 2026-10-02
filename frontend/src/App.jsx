import React, { useState, useEffect } from 'react';
import {
  Compass,
  FileText,
  ShieldCheck,
  Search,
  Share2,
  Activity,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
  Layers,
  Sparkles,
  ArrowRight,
  Database,
  Lock
} from 'lucide-react';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('discovery');
  const [backendStatus, setBackendStatus] = useState('Checking...');
  const [loading, setLoading] = useState(false);
  const [notification, setNotification] = useState(null);

  // Discovery State
  const [query, setQuery] = useState('What are the major causes of antibiotic resistance?');
  const [objective, setObjective] = useState('Identify biological, agricultural, and clinical drivers of AMR');
  const [preferences, setPreferences] = useState('who.int, cdc.gov, lancet');
  const [discoveredSources, setDiscoveredSources] = useState([]);

  // Documents & Chunks State
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);

  // Extraction State
  const [extractionResult, setExtractionResult] = useState(null);

  // Credibility State
  const [credibilityAssessments, setCredibilityAssessments] = useState([]);

  // Retrieval State
  const [retrievalQuery, setRetrievalQuery] = useState('agricultural subtherapeutic antibiotic use');
  const [retrievalResults, setRetrievalResults] = useState([]);

  // Graph State
  const [graphData, setGraphData] = useState(null);

  // Evaluation State
  const [evalReport, setEvalReport] = useState(null);

  // Check Backend Health
  useEffect(() => {
    fetch('/health')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'healthy') {
          setBackendStatus('Connected (100% Free Local Stack)');
        } else {
          setBackendStatus('Degraded');
        }
      })
      .catch(() => setBackendStatus('Offline (Start backend on :8000)'));
  }, []);

  const showNotification = (msg) => {
    setNotification(msg);
    setTimeout(() => setNotification(null), 4000);
  };

  const handlePresetSelect = (q, obj, pref) => {
    setQuery(q);
    setObjective(obj);
    setPreferences(pref);
    setRetrievalQuery(q.split(' ').slice(0, 4).join(' '));
  };

  // 1. Run Source Discovery
  const handleDiscoverSources = async () => {
    setLoading(true);
    try {
      const prefsList = preferences.split(',').map(s => s.trim()).filter(Boolean);
      const res = await fetch('/api/sources/discover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_id: 'task-' + Date.now(),
          query: query,
          research_objective: objective,
          source_preferences: prefsList,
          max_results: 5
        })
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setDiscoveredSources(data);
      showNotification(`Discovered ${data.length} sources`);
    } catch (err) {
      showNotification(`Discovery error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 2. Read & Chunk Documents
  const handleReadDocuments = async () => {
    if (discoveredSources.length === 0) {
      showNotification('Discover or add sources first!');
      return;
    }
    setLoading(true);
    try {
      const res = await fetch('/api/documents/read', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(discoveredSources)
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const docs = await res.json();
      setDocuments(docs);
      if (docs.length > 0) setSelectedDocId(docs[0].id);
      showNotification(`Acquired and chunked ${docs.length} documents`);
    } catch (err) {
      showNotification(`Read error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 3. Extract Evidence
  const handleExtractEvidence = async () => {
    if (documents.length === 0) {
      showNotification('Read documents first before extracting evidence!');
      return;
    }
    setLoading(true);
    try {
      const res = await fetch('/api/evidence/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documents: documents,
          objective: objective,
          session_id: 'sess-' + Date.now()
        })
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setExtractionResult(data);
      showNotification(`Extracted ${data.claims.length} claims with ${data.verified_quote_count} verified quotes`);
    } catch (err) {
      showNotification(`Extraction error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 4. Assess Credibility
  const handleAssessCredibility = async () => {
    if (discoveredSources.length === 0) {
      showNotification('Discover sources first!');
      return;
    }
    setLoading(true);
    try {
      const res = await fetch('/api/credibility/assess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(discoveredSources)
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setCredibilityAssessments(data);
      showNotification(`Assessed credibility for ${data.length} sources`);
    } catch (err) {
      showNotification(`Credibility error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 5. Retrieve Evidence (RAG)
  const handleRetrieveEvidence = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/evidence/retrieve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: retrievalQuery,
          top_k: 5
        })
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const hits = await res.json();
      setRetrievalResults(hits);
      showNotification(`Retrieved ${hits.length} evidence chunks`);
    } catch (err) {
      showNotification(`Retrieval error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 6. Fetch Evidence Graph
  const handleFetchGraph = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/sessions/global/graph');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setGraphData(data);
      showNotification(`Loaded graph with ${data.nodes.length} nodes and ${data.edges.length} edges`);
    } catch (err) {
      showNotification(`Graph error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 7. Run Benchmark Evaluation
  const handleRunEvaluation = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/evaluate', { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setEvalReport(data);
      showNotification(`Benchmark completed across ${data.total_test_cases} test cases!`);
    } catch (err) {
      showNotification(`Eval error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const selectedDoc = documents.find(d => d.id === selectedDocId);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-title">
          <Compass className="text-blue-400" size={26} color="#38bdf8" />
          <div>
            <h1>Autonomous Research Intelligence & Evidence</h1>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Subsystem Test & Exploration Console (100% Free Local AI Stack)
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span className={`badge ${backendStatus.includes('Connected') ? 'badge-green' : 'badge-amber'}`}>
            ● {backendStatus}
          </span>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="btn-secondary"
          >
            Swagger API Docs <ExternalLink size={14} />
          </a>
        </div>
      </header>

      {/* Tabs */}
      <nav className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'discovery' ? 'active' : ''}`}
          onClick={() => setActiveTab('discovery')}
        >
          <Search size={16} /> 1. Source Discovery & Reading
        </button>
        <button
          className={`tab-btn ${activeTab === 'credibility' ? 'active' : ''}`}
          onClick={() => { setActiveTab('credibility'); if (credibilityAssessments.length === 0) handleAssessCredibility(); }}
        >
          <ShieldCheck size={16} /> 2. Source Credibility
        </button>
        <button
          className={`tab-btn ${activeTab === 'extraction' ? 'active' : ''}`}
          onClick={() => setActiveTab('extraction')}
        >
          <FileText size={16} /> 3. Evidence & Grounding
        </button>
        <button
          className={`tab-btn ${activeTab === 'retrieval' ? 'active' : ''}`}
          onClick={() => setActiveTab('retrieval')}
        >
          <Database size={16} /> 4. Hybrid RAG Search
        </button>
        <button
          className={`tab-btn ${activeTab === 'graph' ? 'active' : ''}`}
          onClick={() => { setActiveTab('graph'); if (!graphData) handleFetchGraph(); }}
        >
          <Share2 size={16} /> 5. Evidence Graph
        </button>
        <button
          className={`tab-btn ${activeTab === 'eval' ? 'active' : ''}`}
          onClick={() => setActiveTab('eval')}
        >
          <Activity size={16} /> 6. Evaluation Benchmark
        </button>
      </nav>

      {/* Notification Toast */}
      {notification && (
        <div style={{
          position: 'fixed',
          bottom: '2rem',
          right: '2rem',
          background: 'var(--accent-blue)',
          color: '#0f172a',
          padding: '0.75rem 1.25rem',
          borderRadius: '0.5rem',
          fontWeight: '600',
          boxShadow: '0 10px 15px -3px rgba(0,0,0,0.3)',
          zIndex: 1000
        }}>
          {notification}
        </div>
      )}

      {/* Main Content */}
      <main className="main-content">
        {/* Preset Query Chips */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <Sparkles size={16} color="#fbbf24" />
            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Domain-Independent Research Presets:
            </span>
          </div>
          <div className="presets-container">
            <button
              className="preset-chip"
              onClick={() => handlePresetSelect(
                'What are the major causes of antibiotic resistance?',
                'Identify biological, agricultural, and clinical drivers of AMR',
                'who.int, cdc.gov, thelancet.com'
              )}
            >
              💊 Antibiotic Resistance Causes (Public Health)
            </button>
            <button
              className="preset-chip"
              onClick={() => handlePresetSelect(
                'How has renewable energy adoption changed in India?',
                'Examine solar capacity growth, reverse auctions, and grid challenges',
                'mnre.gov.in, iea.org'
              )}
            >
              ☀️ Renewable Energy Adoption in India (Energy & Policy)
            </button>
            <button
              className="preset-chip"
              onClick={() => handlePresetSelect(
                'What are the security risks of large language models?',
                'Analyze prompt injection, indirect prompt injection, and data exfiltration',
                'owasp.org, arxiv.org'
              )}
            >
              🛡️ LLM Security Risks & Prompt Injection (AI Cybersecurity)
            </button>
          </div>
        </div>

        {/* TAB 1: Source Discovery & Reading */}
        {activeTab === 'discovery' && (
          <div>
            <div className="grid-2">
              <div className="card">
                <div className="card-header">
                  <div className="card-title"><Compass size={18} /> Research Task Specification</div>
                </div>
                <div className="form-group">
                  <label className="form-label">Research Query</label>
                  <input
                    className="form-input"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    placeholder="e.g. Causes of antibiotic resistance"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Research Objective</label>
                  <textarea
                    className="form-textarea"
                    rows={2}
                    value={objective}
                    onChange={e => setObjective(e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Domain/Source Preferences (Optional)</label>
                  <input
                    className="form-input"
                    value={preferences}
                    onChange={e => setPreferences(e.target.value)}
                    placeholder="e.g. who.int, nature.com"
                  />
                </div>
                <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.25rem' }}>
                  <button
                    className="btn-primary"
                    onClick={handleDiscoverSources}
                    disabled={loading}
                  >
                    <Search size={16} /> Discover Sources (Free DDG/Tavily)
                  </button>
                  {discoveredSources.length > 0 && (
                    <button
                      className="btn-secondary"
                      onClick={handleReadDocuments}
                      disabled={loading}
                    >
                      <Layers size={16} /> Read & Chunk All ({discoveredSources.length})
                    </button>
                  )}
                </div>
              </div>

              {/* Discovered Sources List */}
              <div className="card">
                <div className="card-header">
                  <div className="card-title">
                    <FileText size={18} /> Discovered Sources ({discoveredSources.length})
                  </div>
                  <span className="badge badge-amber">Snippet != Evidence</span>
                </div>
                {discoveredSources.length === 0 ? (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                    Click "Discover Sources" to search public web sources. Search snippets are strictly treated as triage info.
                  </p>
                ) : (
                  <div style={{ maxHeight: '380px', overflowY: 'auto' }}>
                    {discoveredSources.map((src, idx) => (
                      <div key={src.id || idx} className="item-card">
                        <div className="item-title">
                          <span style={{ fontSize: '0.95rem' }}>{src.title}</span>
                          <span className="badge badge-blue">{src.source_type}</span>
                        </div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--accent-blue)', marginBottom: '0.3rem' }}>
                          <a href={src.url} target="_blank" rel="noreferrer" style={{ color: 'inherit', textDecoration: 'none' }}>
                            {src.url.substring(0, 60)}... <ExternalLink size={12} style={{ display: 'inline' }} />
                          </a>
                        </div>
                        <p className="item-desc">
                          {src.metadata?.discovery_snippet || src.snippet || 'No discovery snippet'}
                        </p>
                        <div className="provenance-tag">
                          Publisher: {src.publisher || 'Unknown'} | Invariant: is_verified_evidence = false
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Document Chunks Inspector */}
            {documents.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <div className="card-title">
                    <Layers size={18} /> Deterministic Document Chunks ({documents.reduce((acc, d) => acc + d.chunks.length, 0)} total)
                  </div>
                  <button
                    className="btn-primary"
                    onClick={() => { setActiveTab('extraction'); handleExtractEvidence(); }}
                  >
                    Proceed to Evidence Extraction <ArrowRight size={14} />
                  </button>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
                  {documents.map((d, i) => (
                    <button
                      key={d.id}
                      className={`btn-secondary ${selectedDocId === d.id ? 'active' : ''}`}
                      onClick={() => setSelectedDocId(d.id)}
                    >
                      Doc {i + 1}: {d.title.substring(0, 25)}... ({d.chunks.length} chunks)
                    </button>
                  ))}
                </div>

                {selectedDoc && (
                  <div className="grid-2">
                    {selectedDoc.chunks.map((chk, i) => (
                      <div key={chk.id} className="item-card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                          <span className="badge badge-purple">{chk.section || `Chunk #${chk.chunk_index}`}</span>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            {chk.text.length} chars
                          </span>
                        </div>
                        <p style={{ fontSize: '0.85rem', color: '#e2e8f0', marginBottom: '0.5rem' }}>
                          "{chk.text}"
                        </p>
                        <div className="provenance-tag">
                          Provenance: {chk.id.substring(0, 16)}... → {selectedDoc.source_id.substring(0, 12)}...
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* TAB 2: Source Credibility */}
        {activeTab === 'credibility' && (
          <div className="card">
            <div className="card-header">
              <div className="card-title"><ShieldCheck size={18} /> Transparent 7-Factor Source Credibility</div>
              <button className="btn-secondary" onClick={handleAssessCredibility} disabled={loading}>
                Re-assess Sources
              </button>
            </div>
            {credibilityAssessments.length === 0 ? (
              <p style={{ color: 'var(--text-muted)' }}>No sources assessed yet. Discover sources in Tab 1 first.</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {credibilityAssessments.map(c => {
                  const src = discoveredSources.find(s => s.id === c.source_id);
                  return (
                    <div key={c.id} className="item-card" style={{ padding: '1.25rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                        <div>
                          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>{src?.title || c.source_id}</h3>
                          <a href={src?.url} target="_blank" rel="noreferrer" style={{ fontSize: '0.8rem', color: 'var(--accent-blue)' }}>
                            {src?.url}
                          </a>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <span style={{ fontSize: '1.6rem', fontWeight: 700, color: c.overall_score >= 0.8 ? 'var(--accent-green)' : 'var(--accent-amber)' }}>
                            {(c.overall_score * 100).toFixed(0)}%
                          </span>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Overall Credibility</div>
                        </div>
                      </div>

                      {/* Factor Scores Grid */}
                      <div className="grid-3" style={{ marginBottom: '1rem' }}>
                        <div className="metric-box">
                          <div className="metric-label">Authority</div>
                          <div className="metric-val" style={{ fontSize: '1.2rem' }}>{(c.authority_score * 100).toFixed(0)}%</div>
                        </div>
                        <div className="metric-box">
                          <div className="metric-label">Primary Source</div>
                          <div className="metric-val" style={{ fontSize: '1.2rem' }}>{(c.primary_source_score * 100).toFixed(0)}%</div>
                        </div>
                        <div className="metric-box">
                          <div className="metric-label">Recency</div>
                          <div className="metric-val" style={{ fontSize: '1.2rem' }}>{(c.recency_score * 100).toFixed(0)}%</div>
                        </div>
                        <div className="metric-box">
                          <div className="metric-label">Evidence Quality</div>
                          <div className="metric-val" style={{ fontSize: '1.2rem' }}>{(c.evidence_quality_score * 100).toFixed(0)}%</div>
                        </div>
                        <div className="metric-box">
                          <div className="metric-label">Corroboration</div>
                          <div className="metric-val" style={{ fontSize: '1.2rem' }}>{(c.corroboration_score * 100).toFixed(0)}%</div>
                        </div>
                        <div className="metric-box">
                          <div className="metric-label">Peer Review</div>
                          <div className="metric-val" style={{ fontSize: '1.2rem' }}>
                            {c.peer_review_score !== null ? `${(c.peer_review_score * 100).toFixed(0)}%` : 'N/A'}
                          </div>
                        </div>
                      </div>

                      {/* Detailed Reasoning */}
                      <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '0.4rem' }}>
                        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.3rem' }}>
                          Transparent Audit Reasoning:
                        </div>
                        <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.8rem', color: '#cbd5e1', fontFamily: 'inherit' }}>
                          {c.reasoning}
                        </pre>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: Evidence Extraction & Grounding */}
        {activeTab === 'extraction' && (
          <div className="card">
            <div className="card-header">
              <div className="card-title"><FileText size={18} /> Evidence Grounding Verification</div>
              <button className="btn-primary" onClick={handleExtractEvidence} disabled={loading}>
                Run Extraction on Read Documents
              </button>
            </div>
            {!extractionResult ? (
              <p style={{ color: 'var(--text-muted)' }}>
                No extraction run yet. Click "Run Extraction on Read Documents".
              </p>
            ) : (
              <div>
                <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
                  <div className="metric-box">
                    <div className="metric-label">Total Claims Extracted</div>
                    <div className="metric-val">{extractionResult.claims.length}</div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Verified Verbatim Quotes</div>
                    <div className="metric-val" style={{ color: 'var(--accent-green)' }}>
                      {extractionResult.verified_quote_count}
                    </div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Rejected Hallucinations</div>
                    <div className="metric-val" style={{ color: 'var(--accent-red)' }}>
                      {extractionResult.rejected_hallucinated_quotes}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {extractionResult.claims.map((claim, idx) => {
                    const ev = extractionResult.evidence_records.find(e => e.claim_id === claim.id);
                    return (
                      <div key={claim.id || idx} className="item-card">
                        <div className="item-title">
                          <span>Claim #{idx + 1}: {claim.text}</span>
                          <span className="badge badge-green">
                            <CheckCircle2 size={12} style={{ display: 'inline', marginRight: '4px' }} />
                            Grounded ({(claim.confidence * 100).toFixed(0)}%)
                          </span>
                        </div>
                        {ev && (
                          <div>
                            <div className="quote-box">
                              "{ev.text}"
                            </div>
                            <div className="provenance-tag">
                              Location: {ev.location} | Chunk: {ev.chunk_id} | Document: {ev.document_id}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: Hybrid RAG Retrieval */}
        {activeTab === 'retrieval' && (
          <div className="card">
            <div className="card-header">
              <div className="card-title"><Database size={18} /> Hybrid RAG Retriever (Dense Vector + BM25 + RRF)</div>
            </div>
            <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <input
                className="form-input"
                style={{ flex: 1 }}
                value={retrievalQuery}
                onChange={e => setRetrievalQuery(e.target.value)}
                placeholder="Query indexed evidence chunks..."
              />
              <button className="btn-primary" onClick={handleRetrieveEvidence} disabled={loading}>
                <Search size={16} /> Retrieve Evidence
              </button>
            </div>

            {retrievalResults.length === 0 ? (
              <p style={{ color: 'var(--text-muted)' }}>
                Index documents by clicking "Read & Chunk" in Tab 1, then search above.
              </p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {retrievalResults.map((hit, idx) => (
                  <div key={hit.chunk_id || idx} className="item-card">
                    <div className="item-title">
                      <span style={{ fontSize: '0.95rem' }}>Hit #{idx + 1} ({hit.retrieval_method})</span>
                      <span className="badge badge-blue">Relevance: {(hit.relevance_score * 100).toFixed(1)}%</span>
                    </div>
                    <p style={{ fontSize: '0.9rem', color: '#e2e8f0', margin: '0.4rem 0' }}>
                      "{hit.text}"
                    </p>
                    <div className="provenance-tag">
                      Source: {hit.source_url || hit.source_id} | Page: {hit.page_number || 'N/A'} | Chunk: {hit.chunk_id}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 5: Evidence Graph */}
        {activeTab === 'graph' && (
          <div className="card">
            <div className="card-header">
              <div className="card-title"><Share2 size={18} /> Evidence Graph Representation</div>
              <button className="btn-secondary" onClick={handleFetchGraph} disabled={loading}>
                Refresh Graph
              </button>
            </div>
            {!graphData ? (
              <p style={{ color: 'var(--text-muted)' }}>Click "Refresh Graph" to query graph topology.</p>
            ) : (
              <div>
                <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                  <div className="metric-box">
                    <div className="metric-label">Graph Nodes</div>
                    <div className="metric-val">{graphData.nodes.length}</div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Graph Edges (Typed Relations)</div>
                    <div className="metric-val">{graphData.edges.length}</div>
                  </div>
                </div>

                <div className="grid-2">
                  {/* Nodes */}
                  <div className="card" style={{ background: 'rgba(0,0,0,0.2)' }}>
                    <div style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.75rem' }}>
                      Nodes ({graphData.nodes.length})
                    </div>
                    <div style={{ maxHeight: '350px', overflowY: 'auto' }}>
                      {graphData.nodes.map(n => (
                        <div key={n.id} className="item-card" style={{ padding: '0.6rem 0.8rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>{n.label}</span>
                            <span className="badge badge-purple">{n.type}</span>
                          </div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ID: {n.id}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Edges */}
                  <div className="card" style={{ background: 'rgba(0,0,0,0.2)' }}>
                    <div style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.75rem' }}>
                      Typed Relationships / Edges ({graphData.edges.length})
                    </div>
                    <div style={{ maxHeight: '350px', overflowY: 'auto' }}>
                      {graphData.edges.map((e, idx) => (
                        <div key={e.id || idx} className="item-card" style={{ padding: '0.6rem 0.8rem' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}>
                            <span style={{ color: 'var(--accent-blue)' }}>{e.source.substring(0, 10)}...</span>
                            <span className="badge badge-amber">{e.relation}</span>
                            <span style={{ color: 'var(--accent-green)' }}>{e.target.substring(0, 10)}...</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 6: Evaluation Benchmark */}
        {activeTab === 'eval' && (
          <div className="card">
            <div className="card-header">
              <div className="card-title"><Activity size={18} /> Automated Evaluation Benchmark</div>
              <button className="btn-primary" onClick={handleRunEvaluation} disabled={loading}>
                Run Benchmark Suite
              </button>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
              Evaluates retrieval metrics (Recall@3, Precision@3, MRR), evidence coverage, quote verification, and security injection rejection across 3 domain test cases (Public Health AMR, Energy in India, AI/LLM Security).
            </p>

            {evalReport && (
              <div>
                <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
                  <div className="metric-box">
                    <div className="metric-label">Mean Recall@3</div>
                    <div className="metric-val">{(evalReport.mean_recall_at_3 * 100).toFixed(1)}%</div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Mean Precision@3</div>
                    <div className="metric-val">{(evalReport.mean_precision_at_3 * 100).toFixed(1)}%</div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Mean MRR</div>
                    <div className="metric-val">{evalReport.mean_mrr.toFixed(2)}</div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Evidence Coverage</div>
                    <div className="metric-val">{(evalReport.mean_evidence_coverage * 100).toFixed(1)}%</div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Claim Support Rate</div>
                    <div className="metric-val" style={{ color: 'var(--accent-green)' }}>
                      {(evalReport.mean_claim_support_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="metric-box">
                    <div className="metric-label">Security Rejection Rate</div>
                    <div className="metric-val" style={{ color: 'var(--accent-green)' }}>
                      {(evalReport.overall_security_rejection_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {evalReport.case_results.map(c => (
                    <div key={c.case_id} className="item-card">
                      <div className="item-title">
                        <span>{c.domain}</span>
                        <span className="badge badge-green">Passed</span>
                      </div>
                      <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '0.4rem' }}>
                        Query: "{c.query}"
                      </p>
                      <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        <span>Recall@3: {(c.retrieval_metrics.recall_at_3 * 100).toFixed(0)}%</span>
                        <span>Precision@3: {(c.retrieval_metrics.precision_at_3 * 100).toFixed(0)}%</span>
                        <span>MRR: {c.retrieval_metrics.mrr.toFixed(2)}</span>
                        <span>Verified Quotes: {c.evidence_metrics.verified_quotes_count}</span>
                        <span>Rejected Fake Quotes: {c.evidence_metrics.rejected_hallucinations_count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
