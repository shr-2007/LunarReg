export function backendUrl(path: string) {
  const configured = process.env.LUNARREG_API_HOSTPORT || "127.0.0.1:8000";
  const base = configured.startsWith("http://") || configured.startsWith("https://")
    ? configured
    : `http://${configured}`;
  return `${base.replace(/\/$/, "")}${path}`;
}

export function forwardResponse(upstream: Response) {
  const headers = new Headers();
  for (const name of ["content-type", "content-length", "content-disposition", "cache-control"]) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers });
}
