import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || "http://backend:5058";
  
  console.log("Full Destination URL:", `${backendUrl}/move`); 

  try {
    const body = await req.json();
    const response = await fetch(`${backendUrl}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    
    const text = await response.text();

    console.log("Received response from serve.py /move", {
      status: response.status
    });

    const contentType =
      response.headers.get("content-type") || "application/json";

    return new NextResponse(text, {
      status: response.status,
      headers: {
        "Content-Type": contentType
      }
    });
  } catch (err) {
    console.error("FETCH ERROR DETAIL:", err);
    return NextResponse.json({ error: "Connection to Python failed" }, { status: 500 });
  }
}
