import { backendUrl, forwardResponse } from "@/lib/backend";

export async function GET(
  _request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  if (!path.length || path.some((part) => part === ".." || part.includes("/"))) {
    return Response.json({ detail: "Invalid result path." }, { status: 400 });
  }
  try {
    const upstream = await fetch(backendUrl(`/result/${path.map(encodeURIComponent).join("/")}`));
    return forwardResponse(upstream);
  } catch {
    return Response.json({ detail: "Result is unavailable or has expired." }, { status: 404 });
  }
}
