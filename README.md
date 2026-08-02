
Description

Business Context

A large-scale e-commerce business platform operating across 30+ countries, serving over 50 million active customers and employing 2,000+ human support agents who run 24/7 across the US, Europe, India, and Southeast Asia, across product categories including electronics, fashion, groceries, and home appliances. The e-commerce business processes over 200,000 orders per day. With every order comes the possibility of a delivery delay, a payment failure, a wrong item, or a return request. Customers reach out to the support team through tickets to report these issues and expect a fast, accurate resolution.

Every order carries a risk of delivery delays, payment failures, wrong items, and returns, all of which generate support tickets. These tickets are written in highly unstructured ways: some are overloaded with background details where the real issue is buried, others use abbreviations, shorthand, and order codes that are hard to interpret, and some contain so little information that the issue is unclear. As a result:
• Human agents spend the first 2–3 minutes on each ticket just trying to decode what the customer is asking before any resolution work can begin.
• During peak periods (sales, holidays, logistics disruptions), daily ticket volume can spike from ~5,000 to ~15,000, multiplying this inefficiency. High volume and inconsistent ticket content contribute to human agent fatigue and higher error rates when accuracy is most critical.
• Drafting responses is manual, slow, and inconsistent in tone and clarity across human agents, leading to suboptimal customer experiences.

Objective

The objective is to build a POC of an AI-powered ticket intelligence system for the customer support operations of ShopNest Global that:

• Reads incoming customer support tickets in their raw, unstructured form - including overloaded, jargon-filled, and incomplete submissions - and produces a clean, concise summary that gives the support agent an accurate understanding of the customer's issue on first read.
• Generates a professional, empathetic response to the customer that acknowledges their issue and communicates a clear next step, enabling human agents to have a draft response with consistent response quality for their review.
• Evaluates both the generated summary and the generated response automatically using an LLM-as-a-Judge framework, scoring each output against task-specific criteria to ensure quality is measurable, consistent, and auditable at every stage of the pipeline.
The end goal is to demonstrate that AI-assisted summarisation, response generation, and automated evaluation can meaningfully improve the consistency and quality of customer support operations at scale, sufficient to justify broader adoption across ShopNest's global support teams.

Data Dictionary

The data contains the different attributes of the various products and stores.
• support_ticket_id: Unique identifier of the support ticket
• support_ticket_desc: Description of the support ticket

