import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function POST(req: NextRequest) {
  const backendUrl =
    process.env.BACKEND_URL ||
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    "http://backend:5058";

  console.log("Full Destination URL:", `${backendUrl}/move`);

  let body;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON body" },
      { status: 400 }
    );
  }

  const maxAttempts = 10;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const response = await fetch(`${backendUrl}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });

      const text = await response.text();

      console.log("Received response from serve.py /move", {
        status: response.status,
        attempt
      });

      if (response.ok) {
        const contentType =
          response.headers.get("content-type") || "application/json";

        return new NextResponse(text, {
          status: response.status,
          headers: {
            "Content-Type": contentType
          }
        });
      }

      if (response.status >= 500) {
        await sleep(2000);
        continue;
      }

      return NextResponse.json(
        { error: "Backend returned an error" },
        { status: response.status }
      );
    } catch (err) {
      console.error("Fetch attempt failed", { attempt, err });
      await sleep(2000);
    }
  }

  return NextResponse.json(
    {
      error: "Chess engine is starting. Please wait a few seconds and try again."
    },
    { status: 503 }
  );
}
