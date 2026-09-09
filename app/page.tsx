"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowDownToLine, ArrowUpRight, CheckCircle2, ChevronRight,
  CircleDot, Database, FileImage, FlaskConical, GitFork, Gauge, ImagePlus,
  Layers3, LoaderCircle, LockKeyhole, MapPinned, Moon, Orbit, Play,
  RefreshCw, ScanLine, Settings2, Sparkles, Target, UploadCloud, X, Zap,
} from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

type ResultPayload = {
  job_id: string;
  status: "success" | "warning";
  message: string;
  model: string;
  metrics: Record<string, number | string | boolean | null>;
  matrix: number[][];
  assets: Record<"registered" | "reference" | "overlay" | "checkerboard" | "matches", string>;
  download: string;
};

type UploadCardProps = {
  eyebrow: string;
  title: string;
  hint: string;
  file: File | null;
  onFile: (file: File | null) => void;
};

const SOURCES = [
  { mission: "Chandrayaan-2", product: "OHRC · TMC-2 · IIRS", owner: "ISRO / ISSDC", note: "Login required for downloads", href: "https://chmapbrowse.issdc.gov.in/", status: "Source imagery" },
  { mission: "LRO Camera", product: "NAC curated products", owner: "NASA / ASU", note: "Reference image archive", href: "https://www.lroc.asu.edu/images/downloads", status: "Reference imagery" },
  { mission: "LROC QuickMap", product: "2D / 3D lunar browser", owner: "NASA / ASU", note: "Search by coordinates and layers", href: "https://quickmap.lroc.asu.edu/", status: "Discovery tool" },
  { mission: "KAGUYA (SELENE)", product: "TC · MI · ortho products", owner: "JAXA / DARTS", note: "Open PDS lunar archive", href: "https://darts.isas.jaxa.jp/app/pdap/selene/", status: "Reference imagery" },
];

function proxyAsset(path: string) {
  return path.replace(/^\/result\//, "/api/results/");
}

function metric(metrics: ResultPayload["metrics"], key: string, digits = 2) {
  const value = metrics[key];
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function useObjectUrl(file: File | null) {
  const url = useMemo(
    () => file && file.type.startsWith("image/") ? URL.createObjectURL(file) : null,
    [file],
  );
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  return url;
}

function UploadCard({ eyebrow, title, hint, file, onFile }: UploadCardProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const preview = useObjectUrl(file);
  const [dragging, setDragging] = useState(false);

  function acceptFile(next?: File) {
    if (next) onFile(next);
  }

  return (
    <div
      className={`upload-card group ${dragging ? "is-dragging" : ""} ${file ? "has-file" : ""}`}
      onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files[0]); }}
    >
      <input
        ref={inputRef}
        className="sr-only"
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff,.jp2,.img,image/*"
        onChange={(event) => acceptFile(event.target.files?.[0])}
        aria-label={`Upload ${title}`}
      />
      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img className="upload-preview" src={preview} alt={`Preview of ${file?.name}`} />
      ) : null}
      <div className="upload-vignette" />
      <div className="relative z-10 flex h-full flex-col justify-between gap-6 p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="eyebrow">{eyebrow}</p>
            <h3 className="mt-1 text-base font-semibold text-white">{title}</h3>
          </div>
          {file ? (
            <Button size="icon-sm" variant="ghost" className="text-slate-300 hover:bg-white/10 hover:text-white" onClick={() => onFile(null)} aria-label={`Remove ${title}`}>
              <X />
            </Button>
          ) : <div className="icon-well"><ImagePlus /></div>}
        </div>
        {file ? (
          <div className="file-chip">
            <FileImage className="size-4 shrink-0 text-cyan-300" />
            <div className="min-w-0"><p className="truncate text-sm font-medium text-white">{file.name}</p><p className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB · ready</p></div>
          </div>
        ) : (
          <button className="upload-action" type="button" onClick={() => inputRef.current?.click()}>
            <UploadCloud className="size-5" />
            <span><strong>Drop image or browse</strong><small>{hint}</small></span>
          </button>
        )}
      </div>
    </div>
  );
}

function ResultImage({ src, alt }: { src: string; alt: string }) {
  return (
    <div className="result-image-shell">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={proxyAsset(src)} alt={alt} className="result-image" />
      <div className="result-coordinates"><span>REF FRAME</span><span>PIXEL SPACE</span></div>
    </div>
  );
}

function EmptyViewport({ processing, progress }: { processing: boolean; progress: number }) {
  return (
    <div className="empty-viewport">
      <div className={`radar-disc ${processing ? "is-processing" : ""}`}>
        <div className="radar-ring ring-a" /><div className="radar-ring ring-b" />
        <div className="radar-cross cross-a" /><div className="radar-cross cross-b" />
        <Moon className="size-8 text-cyan-200" strokeWidth={1.4} />
      </div>
      <div className="max-w-sm text-center">
        <p className="eyebrow justify-center">{processing ? "Registration sequence active" : "Registration viewport"}</p>
        <h3 className="mt-2 text-lg font-semibold text-white">{processing ? "Finding stable lunar correspondences" : "No registration result yet"}</h3>
        <p className="mt-2 text-sm leading-6 text-slate-400">{processing ? "Normalizing illumination, matching features, rejecting outliers, and refining coordinates." : "Upload a source and reference pair, or run the synthetic demonstration."}</p>
        {processing ? (
          <div className="mt-5">
            <Progress value={progress} className="h-1.5 bg-slate-800 [&_[data-slot=progress-indicator]]:bg-cyan-400" />
            <p className="mt-2 font-mono text-[11px] tracking-wider text-cyan-300">PASS {Math.min(5, Math.ceil(progress / 20))} / 5</p>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function Home() {
  const [source, setSource] = useState<File | null>(null);
  const [reference, setReference] = useState<File | null>(null);
  const [matcher, setMatcher] = useState("sift");
  const [model, setModel] = useState("auto");
  const [maxDimension, setMaxDimension] = useState(2200);
  const [subpixel, setSubpixel] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<ResultPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ready = Boolean(source && reference);

  useEffect(() => {
    if (!processing) return;
    const timer = window.setInterval(() => setProgress((current) => Math.min(92, current + Math.max(1, Math.round((96 - current) / 12)))), 430);
    return () => window.clearInterval(timer);
  }, [processing]);

  const primaryMetrics = useMemo(() => {
    if (!result) return [];
    return [
      { label: "Reprojection RMSE", value: `${metric(result.metrics, "reprojection_rmse_px")} px`, icon: Target },
      { label: "Final control points", value: metric(result.metrics, "final_control_point_count", 0), icon: CircleDot },
      { label: "Inlier ratio", value: `${(Number(result.metrics.inlier_ratio ?? 0) * 100).toFixed(1)}%`, icon: Gauge },
      { label: "Grid coverage", value: `${(Number(result.metrics.uniform_grid_coverage ?? 0) * 100).toFixed(1)}%`, icon: MapPinned },
    ];
  }, [result]);

  async function parseResponse(response: Response) {
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(data?.detail || "The registration service could not complete this request.");
    return data as ResultPayload;
  }

  async function runRegistration() {
    if (!source || !reference) return;
    setProcessing(true); setProgress(5); setError(null); setResult(null);
    const form = new FormData();
    form.append("source", source); form.append("reference", reference); form.append("matcher", matcher);
    form.append("model", model); form.append("max_dimension", String(maxDimension)); form.append("subpixel", String(subpixel));
    try {
      setResult(await parseResponse(await fetch("/api/register", { method: "POST", body: form })));
      setProgress(100);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Unexpected registration error."); }
    finally { setProcessing(false); }
  }

  async function runDemo() {
    setProcessing(true); setProgress(5); setError(null); setResult(null);
    try {
      setResult(await parseResponse(await fetch("/api/demo", { method: "POST" })));
      setProgress(100);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not run the demonstration."); }
    finally { setProcessing(false); }
  }

  return (
    <TooltipProvider>
      <main className="site-shell">
        <header className="topbar">
          <a className="brand" href="#workspace" aria-label="LunarReg home">
            <span className="brand-mark"><Orbit /></span>
            <span><strong>LUNARREG</strong><small>SIH 2026 · ISRO</small></span>
          </a>
          <nav className="hidden items-center gap-1 md:flex" aria-label="Primary navigation">
            <a className="nav-link is-active" href="#workspace">Workspace</a><a className="nav-link" href="#datasets">Datasets</a><a className="nav-link" href="#method">Method</a>
          </nav>
          <div className="flex items-center gap-2">
            <Tooltip><TooltipTrigger asChild><Badge variant="outline" className="status-badge hidden sm:inline-flex"><span className="status-dot" /> API ready</Badge></TooltipTrigger><TooltipContent>FastAPI registration engine</TooltipContent></Tooltip>
            <Button variant="outline" size="sm" className="topbar-button" asChild><a href="#deploy"><GitFork /> Deploy</a></Button>
          </div>
        </header>

        <section id="workspace" className="workspace-section">
          <div className="section-intro">
            <div>
              <div className="flex flex-wrap items-center gap-2"><Badge className="cyan-badge"><ScanLine /> Registration lab</Badge><span className="font-mono text-[11px] tracking-[0.18em] text-slate-500">BUILD 1.0.0</span></div>
              <h1>Align lunar imagery with measured confidence.</h1>
              <p>Geometry-aware correspondence for Chandrayaan-2 and cross-mission reference images, built for illumination, viewpoint, and scale change.</p>
            </div>
            <Button variant="outline" className="demo-button" onClick={runDemo} disabled={processing}>{processing ? <LoaderCircle className="animate-spin" /> : <FlaskConical />}Run synthetic demo</Button>
          </div>

          <div className="workspace-grid">
            <aside className="control-column">
              <div className="panel-heading"><div><p className="eyebrow"><Layers3 /> Input pair</p><h2>Mission imagery</h2></div><Badge variant="outline" className="step-badge">01</Badge></div>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                <UploadCard eyebrow="Moving image" title="Source · Chandrayaan-2" hint="OHRC, TMC-2, IIRS · max 50 MB" file={source} onFile={setSource} />
                <UploadCard eyebrow="Fixed image" title="Reference · LRO / SELENE" hint="NAC, TC, ortho · max 50 MB" file={reference} onFile={setReference} />
              </div>

              <div className="settings-panel">
                <div className="panel-heading compact"><div><p className="eyebrow"><Settings2 /> Configuration</p><h2>Registration controls</h2></div><Badge variant="outline" className="step-badge">02</Badge></div>
                <div className="setting-row">
                  <div><label>Feature matcher</label><span>Scale + illumination robust</span></div>
                  <Select value={matcher} onValueChange={(value) => value && setMatcher(value)}><SelectTrigger className="select-control"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="sift">SIFT · CPU</SelectItem><SelectItem value="ensemble">Ensemble fallback</SelectItem></SelectContent></Select>
                </div>
                <div className="setting-row">
                  <div><label>Transform model</label><span>Auto uses robust model selection</span></div>
                  <Select value={model} onValueChange={(value) => value && setModel(value)}><SelectTrigger className="select-control"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="auto">Auto select</SelectItem><SelectItem value="similarity">Similarity</SelectItem><SelectItem value="affine">Affine</SelectItem><SelectItem value="homography">Homography</SelectItem></SelectContent></Select>
                </div>
                <div className="setting-block">
                  <div className="flex items-center justify-between gap-4"><div><label>Matching resolution</label><span>Longest edge</span></div><span className="value-readout">{maxDimension}px</span></div>
                  <Slider className="mt-4 [&_[data-slot=slider-range]]:bg-cyan-400 [&_[data-slot=slider-thumb]]:border-cyan-300" min={1000} max={4000} step={200} value={[maxDimension]} onValueChange={(value) => setMaxDimension(value[0] ?? 2200)} />
                </div>
                <div className="setting-row"><div><label htmlFor="subpixel">Sub-pixel refinement</label><span>Corner + local NCC refinement</span></div><Switch id="subpixel" checked={subpixel} onCheckedChange={setSubpixel} className="data-[state=checked]:bg-cyan-400" /></div>
              </div>

              {error ? <Alert variant="destructive" className="error-alert"><Activity /><AlertTitle>Registration stopped</AlertTitle><AlertDescription>{error}</AlertDescription></Alert> : null}
              <Button className="run-button" size="lg" disabled={!ready || processing} onClick={runRegistration}>{processing ? <LoaderCircle className="animate-spin" /> : <Play />}{processing ? "Registering pair…" : "Run registration"}{!processing ? <ChevronRight className="ml-auto" /> : null}</Button>
              <p className="privacy-note"><LockKeyhole /> Images are processed in memory; generated jobs expire automatically.</p>
            </aside>

            <section className="result-column" aria-live="polite">
              <div className="result-header">
                <div><p className="eyebrow"><Target /> Output console</p><h2>{result ? "Registration result" : "Awaiting image pair"}</h2></div>
                {result ? (
                  <div className="flex items-center gap-2"><Badge className={result.status === "success" ? "success-badge" : "warning-badge"}><CheckCircle2 /> {result.status}</Badge><Button variant="outline" size="sm" className="download-button" asChild><a href={proxyAsset(result.download)} download><ArrowDownToLine /> Export ZIP</a></Button></div>
                ) : <Badge variant="outline" className="step-badge">03</Badge>}
              </div>

              {result ? (
                <>
                  <div className="metric-grid">
                    {primaryMetrics.map((item) => <article className="metric-card" key={item.label}><item.icon /><div><strong>{item.value}</strong><span>{item.label}</span></div></article>)}
                    <article className="metric-card model-card"><Sparkles /><div><strong>{result.model}</strong><span>Selected model</span></div></article>
                  </div>
                  <Tabs defaultValue="overlay" className="result-tabs">
                    <div className="tabs-toolbar"><TabsList className="tabs-list"><TabsTrigger value="overlay">Overlay</TabsTrigger><TabsTrigger value="checkerboard">Checkerboard</TabsTrigger><TabsTrigger value="matches">Matches</TabsTrigger><TabsTrigger value="registered">Registered</TabsTrigger></TabsList><span className="runtime"><Zap /> {metric(result.metrics, "runtime_seconds")}s</span></div>
                    <TabsContent value="overlay"><ResultImage src={result.assets.overlay} alt="False-color registration overlay" /></TabsContent>
                    <TabsContent value="checkerboard"><ResultImage src={result.assets.checkerboard} alt="Registration checkerboard comparison" /></TabsContent>
                    <TabsContent value="matches"><ResultImage src={result.assets.matches} alt="Distributed corresponding match points" /></TabsContent>
                    <TabsContent value="registered"><ResultImage src={result.assets.registered} alt="Registered source image" /></TabsContent>
                  </Tabs>
                  <div className="result-foot"><p><CheckCircle2 /> {result.message}</p><span>JOB {result.job_id.toUpperCase()}</span></div>
                </>
              ) : <EmptyViewport processing={processing} progress={progress} />}
            </section>
          </div>
        </section>

        <section id="datasets" className="content-section">
          <div className="content-heading"><div><p className="eyebrow"><Database /> Dataset access</p><h2>Verified lunar data portals</h2><p>Use overlapping coordinates and similar ground sampling distance where possible. The official SIH pair package remains pending.</p></div><Badge variant="outline" className="pending-badge"><RefreshCw /> SIH pair link · TBD</Badge></div>
          <div className="source-table" role="table" aria-label="Lunar dataset portals">
            <div className="source-row source-head" role="row"><span>Mission / archive</span><span>Products</span><span>Role</span><span>Access</span></div>
            {SOURCES.map((item) => <div className="source-row" role="row" key={item.mission}><div><strong>{item.mission}</strong><small>{item.owner}</small></div><div><strong>{item.product}</strong><small>{item.note}</small></div><Badge variant="outline" className="role-badge">{item.status}</Badge><Button variant="ghost" size="sm" className="source-link" asChild><a href={item.href} target="_blank" rel="noreferrer">Open portal <ArrowUpRight /></a></Button></div>)}
          </div>
        </section>

        <section id="method" className="content-section method-section">
          <div className="content-heading"><div><p className="eyebrow"><Orbit /> Registration method</p><h2>From raw pixels to auditable alignment</h2><p>A deterministic baseline designed to be explainable during judging and extensible to LightGlue or LoFTR when GPU resources are available.</p></div></div>
          <div className="pipeline-grid">
            {[["01","Ingest","PNG, JPEG, TIFF, JP2 and IMG"],["02","Normalize","Robust intensity + gradient views"],["03","Match","Multi-scale SIFT correspondences"],["04","Reject","RANSAC and model selection"],["05","Refine","Sub-pixel corner + local NCC"],["06","Export","Images, points, metrics and matrix"]].map(([step,title,text]) => <article className="pipeline-card" key={step}><span>{step}</span><h3>{title}</h3><p>{text}</p></article>)}
          </div>
        </section>

        <section id="deploy" className="deploy-section">
          <div><p className="eyebrow"><GitFork /> Delivery</p><h2>One repository. One Render Blueprint.</h2><p>Push the project to GitHub, create a Render Blueprint from the repository, and the frontend plus registration API deploy as connected services.</p></div>
          <div className="deploy-command"><span>render.yaml</span><code>2 services · Node UI + FastAPI engine</code></div>
        </section>

        <footer><div className="brand compact-brand"><span className="brand-mark"><Moon /></span><span><strong>LUNARREG</strong><small>SUB-PIXEL LUNAR REGISTRATION</small></span></div><p>Proof-of-concept for Smart India Hackathon 2026.</p></footer>
      </main>
    </TooltipProvider>
  );
}
