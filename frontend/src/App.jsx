import { useEffect, useMemo, useState } from 'react';
import {
  Activity, ArrowUpRight, BookOpen, Check, ChevronRight, CircleAlert,
  Compass, Database, FileCheck2, FileText, FlaskConical, Globe2, LayoutDashboard,
  LoaderCircle, Menu, Network, PanelLeft, Play, Search, Share2, ShieldCheck,
  Sparkles, X,
} from 'lucide-react';
import './App.css';

const presets = [
  { label: 'Antibiotic resistance', question: 'What are the major causes of antibiotic resistance?', description: 'Public health and clinical drivers' },
  { label: 'Renewable energy in India', question: 'How has renewable energy adoption changed in India?', description: 'Capacity growth, policy, and grid challenges' },
  { label: 'LLM security risks', question: 'What are the security risks of large language models?', description: 'Prompt injection and data exfiltration' },
];

const stages = [
  ['query_understanding', 'Understanding the question'],
  ['planning', 'Building a research plan'],
  ['research_execution', 'Gathering and reading sources'],
  ['evidence_verification', 'Checking evidence and provenance'],
  ['research_review', 'Reviewing coverage'],
  ['report_writing', 'Writing the research brief'],
  ['citation_validation', 'Auditing citations'],
];

const percent = (value) => `${Math.round((value || 0) * 100)}%`;
const host = (url) => { try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return 'Unknown source'; } };

export default function App() {
  const [question, setQuestion] = useState(presets[0].question);
  const [iterations, setIterations] = useState(3);
  const [result, setResult] = useState(null);
  const [view, setView] = useState('brief');
  const [running, setRunning] = useState(false);
  const [health, setHealth] = useState('checking');
  const [error, setError] = useState('');
  const [mobileNav, setMobileNav] = useState(false);
  const [supplemental, setSupplemental] = useState({ retrieval: [], graph: null, evaluation: null });

  useEffect(() => {
    fetch('/health').then((response) => response.ok ? response.json() : Promise.reject())
      .then((data) => setHealth(data.status === 'healthy' ? 'online' : 'degraded'))
      .catch(() => setHealth('offline'));
  }, []);

  const runResearch = async (event) => {
    event?.preventDefault();
    if (!question.trim() || running) return;
    setRunning(true); setError(''); setResult(null); setSupplemental({ retrieval: [], graph: null, evaluation: null }); setView('brief');
    try {
      const response = await fetch('/api/orchestration/research', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim(), max_iterations: iterations }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
      const [retrieval, graph, evaluation] = await Promise.allSettled([
        fetch('/api/evidence/retrieve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: question.trim(), top_k: 5 }) }).then((response) => response.ok ? response.json() : []),
        fetch(`/api/sessions/${data.session_id}/graph`).then((response) => response.ok ? response.json() : null),
        fetch('/api/evaluate', { method: 'POST' }).then((response) => response.ok ? response.json() : null),
      ]);
      const extras = {
        retrieval: retrieval.status === 'fulfilled' ? retrieval.value : [],
        graph: graph.status === 'fulfilled' ? graph.value : null,
        evaluation: evaluation.status === 'fulfilled' ? evaluation.value : null,
      };
      setSupplemental(extras);
      setResult(data);
    } catch (requestError) { setError(requestError.message || 'The research run could not be completed.'); }
    finally { setRunning(false); }
  };

  const report = result?.final_report;
  const navItems = [
    ['brief', 'Research brief', LayoutDashboard], ['sources', 'Source discovery', Globe2],
    ['credibility', 'Credibility', ShieldCheck], ['documents', 'Documents & chunks', FileText],
    ['evidence', 'Evidence & metrics', FileCheck2], ['retrieval', 'Hybrid RAG', Database],
    ['graph', 'Evidence graph', Share2], ['evaluation', 'Evaluation', Activity],
    ['plan', 'Agent plan', Network], ['activity', 'Run activity', Activity],
  ];

  return <div className="workspace">
    <header className="topbar">
      <div className="brand-lockup"><div className="brand-mark"><Compass size={19} /></div><div><div className="brand-name">Northstar</div><div className="brand-caption">Evidence research workspace</div></div></div>
      <div className="topbar-actions"><span className={`system-state ${health}`}><span /> {health === 'online' ? 'Pipeline online' : health === 'checking' ? 'Checking pipeline' : 'Pipeline offline'}</span><a className="docs-link" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API docs <ArrowUpRight size={14} /></a><button className="icon-button mobile-menu" aria-label="Toggle navigation" onClick={() => setMobileNav(!mobileNav)}><Menu size={19} /></button></div>
    </header>
    <div className="app-body">
      <aside className={`sidebar ${mobileNav ? 'open' : ''}`}>
        <div className="sidebar-section-label">Workspace</div><div className="sidebar-intro"><Compass size={18} /><strong>{result ? 'Run complete' : 'Ready to research'}</strong><span>{result ? 'Use the result navigation to inspect every pipeline stage.' : 'Start a question to unlock the full research record.'}</span></div>
        <div className="sidebar-rule" /><div className="sidebar-note"><ShieldCheck size={17} /><div><strong>Grounded by default</strong><span>Every reported claim is checked against its source passage.</span></div></div><div className="sidebar-footer"><span className="local-dot" /> Local research stack</div>
      </aside>
      <main className="content">
        <div className="content-heading"><div><span className="eyebrow">Autonomous research</span><h1>{result ? 'Your research brief' : 'What would you like to understand?'}</h1></div>{result && <button className="quiet-button" onClick={() => { setResult(null); setError(''); setView('brief'); }}><Sparkles size={15} /> New research</button>}</div>
        {result && <nav className="result-nav" aria-label="Research result sections">{navItems.map(([key, label, Icon]) => <button key={key} className={view === key ? 'active' : ''} onClick={() => setView(key)}><Icon size={15} /><span>{label}</span></button>)}</nav>}
        {!result && !running && <section className="composer-card"><div className="composer-label"><span className="pulse-dot" /> Ask the research agent</div><form onSubmit={runResearch}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question worth investigating..." rows={4} /><div className="composer-footer"><div className="composer-meta"><FlaskConical size={15} /> Searches, reads, verifies, and cites public sources</div><div className="composer-controls"><label>Depth <select value={iterations} onChange={(event) => setIterations(Number(event.target.value))}><option value={1}>Quick</option><option value={2}>Balanced</option><option value={3}>Thorough</option><option value={4}>Deep</option></select></label><button className="run-button" type="submit"><Play size={16} fill="currentColor" /> Start research</button></div></div></form><div className="preset-row"><span>Try a starting point</span>{presets.map((preset) => <button key={preset.label} className="preset" onClick={() => setQuestion(preset.question)}><span>{preset.label}</span><small>{preset.description}</small></button>)}</div></section>}
        {running && <RunProgress question={question} />}
        {error && <div className="error-banner"><CircleAlert size={18} /><div><strong>Research run failed</strong><span>{error}</span></div><button className="icon-button" onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>}
        {result && view === 'brief' && <><BriefView result={result} report={report} onRunAgain={() => runResearch()} /><RunRecord result={result} /></>}
        {result && view === 'evidence' && <EvidenceView result={result} />}
        {result && view === 'sources' && <SourcesView result={result} />}
        {result && view === 'credibility' && <CredibilityView result={result} />}
        {result && view === 'documents' && <DocumentsView result={result} />}
        {result && view === 'retrieval' && <RetrievalView result={result} hits={supplemental.retrieval} />}
        {result && view === 'graph' && <GraphView graph={supplemental.graph} />}
        {result && view === 'evaluation' && <EvaluationView report={supplemental.evaluation} />}
        {result && view === 'plan' && <PlanView result={result} />}
        {result && view === 'activity' && <ActivityView result={result} />}
        {!result && !running && !error && <div className="empty-hint"><PanelLeft size={17} /> Your completed brief, evidence, and source trail will appear here.</div>}
      </main>
    </div>
  </div>;
}

function RunProgress({ question }) {
  return <section className="progress-card"><div className="progress-top"><div><span className="eyebrow">Agent in motion</span><h2>Researching your question</h2><p>{question}</p></div><LoaderCircle className="spinner" size={28} /></div><div className="stage-list">{stages.map(([key, label], index) => <div className="stage" key={key}><span className="stage-icon"><LoaderCircle size={14} className="stage-spinner" /></span><span>{label}</span>{index < 2 && <small>working</small>}</div>)}</div><div className="progress-foot"><span>This may take a minute while the agent reads and verifies sources.</span><div className="progress-line"><span /></div></div></section>;
}

function BriefView({ result, report, onRunAgain }) {
  if (!report) return <div className="empty-state"><CircleAlert size={22} /><h2>No final report was returned</h2><p>The agent completed with status “{result.status}”. Inspect the plan and activity views for details.</p></div>;
  const verification = result.verification_report;
  const validation = result.citation_validation;
  const supportRate = verification?.total_claims_checked ? verification.supported_claims_count / verification.total_claims_checked : 0;
  return <><div className="result-meta"><span className="success-label"><Check size={14} /> Research complete</span><span>Session {result.session_id}</span><span>{new Date(report.created_at).toLocaleString()}</span><button className="text-button" onClick={onRunAgain}>Run again <ArrowUpRight size={14} /></button></div><section className="report-hero"><div className="report-kicker">Research brief</div><h2>{report.title}</h2><p>{report.executive_summary}</p></section><div className="stat-strip"><Stat label="Sources read" value={result.discovered_sources_count} icon={Globe2} /><Stat label="Claims checked" value={verification?.total_claims_checked || result.extracted_claims_count} icon={FileCheck2} /><Stat label="Supported" value={percent(supportRate)} icon={ShieldCheck} /><Stat label="Citations" value={validation?.valid_citations_count ?? report.references.length} icon={BookOpen} /></div><div className="brief-grid"><div><section className="panel report-section"><SectionHeading eyebrow="Findings" title="What the evidence says" count={report.key_findings.length} />{report.key_findings.map((finding, index) => <article className="finding" key={`${finding.topic_title}-${index}`}><div className="finding-number">0{index + 1}</div><div><h3>{finding.topic_title}</h3><p>{finding.summary}</p>{finding.claims?.map((claim) => <div className="claim-line" key={claim}><Check size={14} />{claim}</div>)}</div></article>)}</section><section className="panel report-section"><SectionHeading eyebrow="Method" title="How this was researched" /><p className="body-copy">{report.research_scope_and_methodology}</p><p className="body-copy">{report.supporting_evidence_summary}</p></section>{report.conclusion && <section className="conclusion"><span className="eyebrow">Bottom line</span><p>{report.conclusion}</p></section>}</div><aside className="right-rail"><TrustCard result={result} /><SourceList sources={result.sources} /><ReviewNotes report={report} contradictions={result.contradictions} /></aside></div></>;
}

function RunRecord({ result }) {
  const query = result.structured_query;
  const verification = result.verification_report;
  const references = result.final_report?.references || [];
  return <section className="run-record">
    <div className="run-record-heading"><div><span className="eyebrow">Complete run record</span><h2>Everything the agent produced</h2></div><span className="record-session">{result.iteration_count} iteration{result.iteration_count === 1 ? '' : 's'}</span></div>
    <div className="record-grid">
      <section className="panel record-panel"><SectionHeading eyebrow="Query understanding" title="Research frame" /><div className="record-field"><span>Objective</span><strong>{query?.research_objective || 'Not returned'}</strong></div><div className="record-field"><span>Scope</span><strong>{query?.scope || 'Not returned'}</strong></div><div className="chip-list">{query?.key_concepts?.map((concept) => <span key={concept}>{concept}</span>)}</div>{query?.sub_questions?.map((subQuestion, index) => <div className="record-question" key={subQuestion}><b>0{index + 1}</b>{subQuestion}</div>)}</section>
      <section className="panel record-panel"><SectionHeading eyebrow="Research plan" title={`${result.plan?.tasks?.length || 0} tasks`} />{result.plan?.tasks?.map((task, index) => <div className="record-task" key={task.task_id}><b>0{index + 1}</b><div><strong>{task.title}</strong><span>{task.objective}</span></div><em>{task.status}</em></div>)}</section>
      <section className="panel record-panel record-wide"><SectionHeading eyebrow="Source discovery and credibility" title={`${result.sources?.length || 0} sources acquired`} count={result.credibility_assessments?.length || 0} />{(result.sources || []).map((source) => { const assessment = result.credibility_assessments?.find((item) => item.source_id === source.id); return <div className="record-source" key={source.id}><span className="source-favicon"><Globe2 size={13} /></span><div><strong>{source.title}</strong><small>{host(source.url)} · {source.publisher || 'Publisher not listed'}</small></div><span className="source-score">{assessment ? percent(assessment.overall_score) : 'Assessed'}</span></div>; })}</section>
      <section className="panel record-panel record-wide"><SectionHeading eyebrow="Document reader and extraction" title={`${result.documents?.length || 0} documents · ${result.claims?.length || 0} claims`} count={result.evidence_records?.length || 0} />{(result.claims || []).length === 0 ? <EmptyPanel text="No extracted claims were returned." /> : result.claims.map((claim) => { const evidence = result.evidence_records?.filter((item) => item.claim_id === claim.id) || []; return <article className="record-claim" key={claim.id}><div><strong>{claim.text}</strong><span>Confidence {percent(claim.confidence)} · {evidence.length} evidence passage{evidence.length === 1 ? '' : 's'}</span></div>{evidence.slice(0, 2).map((item) => <blockquote key={item.id}>“{item.text}” <small>{item.location}</small></blockquote>)}</article>; })}</section>
      <section className="panel record-panel"><SectionHeading eyebrow="Citation audit" title={`${references.length} references`} /><div className={`audit-result ${result.citation_validation?.is_valid ? 'passed' : 'warning'}`}><ShieldCheck size={16} /><strong>{result.citation_validation?.is_valid ? 'All citations validated' : 'Citation warnings found'}</strong></div>{references.map((reference) => <a className="record-reference" href={reference.source_url} target="_blank" rel="noreferrer" key={reference.citation_id}><span>{reference.source_title}</span><ArrowUpRight size={13} /></a>)}</section>
      <section className="panel record-panel"><SectionHeading eyebrow="Run activity" title={`${result.audit_logs?.length || 0} stage events`} />{(result.audit_logs || []).map((event, index) => <div className="record-event" key={`${event.stage}-${index}`}><span className="event-dot" /><div><strong>{event.stage?.replaceAll('_', ' ')}</strong><small>{event.duration_ms} ms</small></div></div>)}</section>
    </div>
  </section>;
}

function EvidenceView({ result }) {
  const verification = result.verification_report; const findings = verification?.claim_findings || [];
  return <div className="view-stack"><ViewTitle eyebrow="Evidence ledger" title="Every claim has a trail" description="Inspect how the agent connected claims to verified passages and source URLs." /><div className="ledger-summary"><Stat label="Verified passages" value={result.verified_evidence_count} icon={ShieldCheck} /><Stat label="Supported claims" value={verification?.supported_claims_count || 0} icon={Check} /><Stat label="Contradictions" value={result.contradictions?.length || 0} icon={CircleAlert} /></div><section className="panel ledger">{findings.length === 0 ? <EmptyPanel text="No claim verification details were returned." /> : findings.map((finding) => <article className="ledger-row" key={finding.claim_id}><div className={`verification-icon ${finding.is_supported ? 'good' : 'warn'}`}>{finding.is_supported ? <Check size={16} /> : <CircleAlert size={16} />}</div><div className="ledger-main"><h3>{finding.claim_text}</h3><p>{finding.reasoning}</p><div className="ledger-tags"><span>{finding.is_supported ? `Supported ${percent(finding.support_strength)}` : 'Needs support'}</span><span>{finding.has_valid_provenance ? 'Provenance verified' : 'Provenance gap'}</span>{finding.flags?.map((flag) => <span key={flag}>{flag}</span>)}</div></div></article>)}</section></div>;
}

function SourcesView({ result }) { return <div className="view-stack"><ViewTitle eyebrow="Source library" title="The material behind the brief" description="Search results are treated as leads. The agent reads the underlying pages before using them as evidence." /><section className="source-grid">{(result.sources || []).map((source) => <article className="source-card" key={source.id}><div className="source-top"><div className="source-favicon"><Globe2 size={16} /></div><span className="source-type">{source.source_type || 'webpage'}</span></div><h3>{source.title}</h3><span className="source-domain">{host(source.url)}</span><p>{source.metadata?.discovery_snippet || 'Source acquired and available for evidence extraction.'}</p><a href={source.url} target="_blank" rel="noreferrer">Open source <ArrowUpRight size={14} /></a></article>)}</section></div>; }

function CredibilityView({ result }) { return <div className="view-stack"><ViewTitle eyebrow="Source credibility" title="Transparent source assessment" description="Each source is scored across authority, primary-source quality, recency, evidence quality, peer review, and corroboration." /><section className="source-assessment-list">{(result.credibility_assessments || []).map((assessment) => { const source = result.sources?.find((item) => item.id === assessment.source_id); return <article className="panel assessment-card" key={assessment.id}><div className="assessment-header"><div><span className="eyebrow">{host(source?.url)}</span><h3>{source?.title || assessment.source_id}</h3></div><strong>{percent(assessment.overall_score)}</strong></div><div className="factor-grid">{[['Authority', assessment.authority_score], ['Primary source', assessment.primary_source_score], ['Recency', assessment.recency_score], ['Evidence quality', assessment.evidence_quality_score], ['Peer review', assessment.peer_review_score], ['Corroboration', assessment.corroboration_score]].map(([label, value]) => <div className="factor" key={label}><div><span>{label}</span><b>{value === null ? 'N/A' : percent(value)}</b></div><i><em style={{ width: value === null ? '0%' : `${value * 100}%` }} /></i></div>)}</div><p className="assessment-reasoning">{assessment.reasoning}</p></article>; })}</section></div>; }

function DocumentsView({ result }) { const documents = result.documents || []; return <div className="view-stack"><ViewTitle eyebrow="Document reader" title="Documents and deterministic chunks" description="The reader preserves page and section provenance before evidence extraction." /><div className="ledger-summary"><Stat label="Documents" value={documents.length} icon={FileText} /><Stat label="Chunks" value={documents.reduce((total, document) => total + (document.chunks?.length || 0), 0)} icon={BookOpen} /><Stat label="Claims extracted" value={result.claims?.length || 0} icon={FileCheck2} /></div><section className="document-list">{documents.map((document) => <article className="panel document-card" key={document.id}><div className="document-heading"><div><span className="eyebrow">{document.document_type || 'document'}</span><h3>{document.title}</h3></div><span className="count-pill">{document.chunks?.length || 0} chunks</span></div><div className="chunk-grid">{(document.chunks || []).map((chunk) => <div className="chunk-card" key={chunk.id}><span>{chunk.section || `Chunk ${chunk.chunk_index + 1}`}</span><p>{chunk.text}</p><small>{chunk.text.length} characters · {chunk.page_number ? `page ${chunk.page_number}` : 'provenance preserved'}</small></div>)}</div></article>)}</section></div>; }

function RetrievalView({ result, hits }) { return <div className="view-stack"><ViewTitle eyebrow="Hybrid RAG" title="Retrieval quality and ranked evidence" description="Dense vector search and BM25 keyword search are fused with reciprocal rank fusion." /><div className="ledger-summary"><Stat label="Retrieved hits" value={hits.length} icon={Database} /><Stat label="Indexed chunks" value={result.documents?.reduce((total, document) => total + (document.chunks?.length || 0), 0)} icon={BookOpen} /><Stat label="Evidence records" value={result.evidence_records?.length || 0} icon={FileCheck2} /></div><section className="panel retrieval-list">{hits.length === 0 ? <EmptyPanel text="No ranked hits were returned for this question." /> : hits.map((hit, index) => <article className="retrieval-row" key={hit.chunk_id || index}><span className="rank">0{index + 1}</span><div><div className="retrieval-heading"><strong>{hit.retrieval_method || 'hybrid retrieval'}</strong><span>{percent(hit.relevance_score)} relevance</span></div><p>{hit.text}</p><small>{hit.source_url || hit.source_id} · {hit.page_number ? `page ${hit.page_number}` : 'source passage'} · {hit.chunk_id}</small></div></article>)}</section></div>; }

function GraphView({ graph }) { return <div className="view-stack"><ViewTitle eyebrow="Evidence graph" title="How the evidence connects" description="Typed relationships expose how sources, documents, claims, and evidence support the final brief." />{!graph ? <div className="empty-state"><CircleAlert size={22} /><h2>Graph data unavailable</h2><p>The research completed, but this session did not return graph topology.</p></div> : <><div className="ledger-summary"><Stat label="Nodes" value={graph.nodes?.length || 0} icon={Network} /><Stat label="Relationships" value={graph.edges?.length || 0} icon={Share2} /><Stat label="Evidence links" value={graph.edges?.filter((edge) => edge.relation === 'supports').length || 0} icon={ShieldCheck} /></div><div className="graph-grid"><section className="panel"><SectionHeading eyebrow="Nodes" title="Research entities" />{graph.nodes?.map((node) => <div className="graph-row" key={node.id}><span className={`node-dot ${node.type}`} /><strong>{node.label}</strong><small>{node.type}</small></div>)}</section><section className="panel"><SectionHeading eyebrow="Edges" title="Typed relationships" />{graph.edges?.map((edge) => <div className="graph-row" key={edge.id}><span className="edge-source">{edge.source.slice(0, 8)}</span><em>{edge.relation}</em><span className="edge-target">{edge.target.slice(0, 8)}</span></div>)}</section></div></>}</div>; }

function EvaluationView({ report }) { return <div className="view-stack"><ViewTitle eyebrow="Evaluation benchmark" title="Pipeline quality checks" description="The benchmark covers retrieval, evidence grounding, claim support, and security rejection across multiple domains." />{!report ? <div className="empty-state"><LoaderCircle className="spinner" size={22} /><h2>Evaluation is still loading</h2><p>The benchmark runs after the agent completes. Refresh this view in a moment.</p></div> : <><div className="stat-strip"><Stat label="Recall @3" value={percent(report.mean_recall_at_3)} icon={Search} /><Stat label="Precision @3" value={percent(report.mean_precision_at_3)} icon={FileCheck2} /><Stat label="Mean MRR" value={(report.mean_mrr || 0).toFixed(2)} icon={Activity} /><Stat label="Security rejection" value={percent(report.overall_security_rejection_rate)} icon={ShieldCheck} /></div><section className="panel evaluation-list">{report.case_results?.map((item) => <article className="evaluation-row" key={item.case_id}><div><span className="eyebrow">{item.domain}</span><h3>{item.query}</h3></div><span className="task-status completed">passed</span><div className="evaluation-metrics"><span>Recall {percent(item.retrieval_metrics.recall_at_3)}</span><span>Precision {percent(item.retrieval_metrics.precision_at_3)}</span><span>MRR {(item.retrieval_metrics.mrr || 0).toFixed(2)}</span><span>Quotes {item.evidence_metrics.verified_quotes_count}</span></div></article>)}</section></>}</div>; }

function PlanView({ result }) { const query = result.structured_query; return <div className="view-stack"><ViewTitle eyebrow="Agent plan" title="How the question was broken down" description="The planner turns one question into focused, auditable research tasks." /><div className="plan-grid"><section className="panel"><SectionHeading eyebrow="Query understanding" title="Research frame" /><div className="frame-row"><span>Objective</span><strong>{query?.research_objective}</strong></div><div className="frame-row"><span>Scope</span><strong>{query?.scope}</strong></div><div className="chip-list">{query?.key_concepts?.map((concept) => <span key={concept}>{concept}</span>)}</div>{query?.sub_questions?.map((subQuestion, index) => <div className="sub-question" key={subQuestion}><span>0{index + 1}</span>{subQuestion}</div>)}</section><section className="panel"><SectionHeading eyebrow="Execution plan" title={`${result.plan?.tasks?.length || 0} research tasks`} />{result.plan?.tasks?.map((task, index) => <div className="task-row" key={task.task_id}><span className="task-number">{index + 1}</span><div><h3>{task.title}</h3><p>{task.objective}</p></div><span className={`task-status ${task.status}`}>{task.status}</span></div>)}</section></div></div>; }

function ActivityView({ result }) { return <div className="view-stack"><ViewTitle eyebrow="Run activity" title="A transparent execution trail" description="A record of the stages the autonomous workflow completed." /><section className="panel activity-list">{(result.audit_logs || []).map((event, index) => <div className="activity-row" key={`${event.stage}-${index}`}><div className="activity-marker"><Check size={13} /></div><div><h3>{event.stage?.replaceAll('_', ' ')}</h3><p>{Object.entries(event).filter(([key]) => !['stage', 'timestamp', 'duration_ms'].includes(key)).map(([key, value]) => `${key.replaceAll('_', ' ')}: ${Array.isArray(value) ? value.length : value}`).join(' · ')}</p></div><time>{event.duration_ms} ms</time></div>)}</section></div>; }

function TrustCard({ result }) { const verification = result.verification_report; const validation = result.citation_validation; const passed = validation?.is_valid && verification?.provenance_integrity_passed; return <section className="trust-card"><div className="trust-icon"><ShieldCheck size={20} /></div><div><span className="eyebrow">Reliability check</span><h3>{passed ? 'Grounding passed' : 'Review warnings'}</h3><p>{passed ? 'Claims and citations trace back to acquired source passages.' : 'Some parts of this brief need closer review.'}</p></div><div className="trust-details"><span><b>{result.verified_evidence_count}</b> verified passages</span><span><b>{validation?.valid_citations_count || 0}</b> valid citations</span></div></section>; }

function SourceList({ sources = [] }) { return <section className="side-section"><SectionHeading eyebrow="Sources" title="Top references" count={sources.length} />{sources.slice(0, 4).map((source) => <a className="side-source" href={source.url} target="_blank" rel="noreferrer" key={source.id}><span className="source-favicon"><Globe2 size={13} /></span><span><strong>{source.title}</strong><small>{host(source.url)}</small></span><ArrowUpRight size={14} /></a>)}{sources.length > 4 && <small className="more-note">+ {sources.length - 4} more in Source library</small>}</section>; }
function ReviewNotes({ report, contradictions = [] }) { const notes = [...(report.limitations_and_uncertainties || [])]; if (contradictions.length) notes.push(`${contradictions.length} contradictory finding(s) were flagged.`); if (!notes.length) return null; return <section className="notes-card"><div className="notes-heading"><CircleAlert size={16} /> Review notes</div>{notes.slice(0, 3).map((note) => <p key={note}>{note}</p>)}</section>; }
function ViewTitle({ eyebrow, title, description }) { return <div className="view-title"><span className="eyebrow">{eyebrow}</span><h2>{title}</h2><p>{description}</p></div>; }
function SectionHeading({ eyebrow, title, count }) { return <div className="section-heading"><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div>{count !== undefined && <span className="count-pill">{count}</span>}</div>; }
function Stat({ label, value, icon: Icon }) { return <div className="stat"><Icon size={16} /><div><strong>{value}</strong><span>{label}</span></div></div>; }
function EmptyPanel({ text }) { return <div className="empty-panel"><Search size={18} /><span>{text}</span></div>; }