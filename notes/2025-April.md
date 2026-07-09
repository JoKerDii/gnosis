---
title: 2025 April - What I Have Read
date: 2025-04-01 10:27:12
tags: readings
---

## Substack

> *A Parquet file is composed of Row Groups, Column Chunk, and Pages.*
>
> *Parquet is a self-described file format that contains all the information needed for the application that consumes the file. This allows the software to efficiently understand and process the file without requiring external information. Thus, the metadata is the crucial part of Parquet. They include Magic Number, FileMetadata, and PageHeader.*
>
> *Google Dremel (the query engine behind BigQuery) inspired Parquet’s approach to implementing nested and repeated field storage. In a [2010 paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36632.pdf) introducing Dremel, Google detailed its method for efficiently handling nested and repeated fields in analytics workloads using definition level (for nested fields) and repetition level (for array-like fields). I wrote an article about this approach seven months ago; you can read it here:*
>
> **― I spent 8 hours learning Parquet. Here’s what I discovered - Vu Trinh** [[Link](https://vutr.substack.com/p/the-overview-of-parquet-file-format)]

> *The overall BigQuery architecture includes independent components for query execution, storage, a container management system, and a shuffler service:*
>
> - ***Colossus**: A distributed storage system that holds and stores data.*
> - ***Dremel**: The distributed query engine.*
> - ***Borg is** Google’s large-scale cluster management system that can reliably manage and orchestrate compute resources. ([Borg is the predecessor of Kubernetes.](https://kubernetes.io/blog/2015/04/borg-predecessor-to-kubernetes/)) We will return to Borg when discussing the Vortex architecture.*
> - ***Dedicate shuffle service**: Dremel was inspired by the map-reduce paradigm to operate and manage the data shuffle between stages efficiently; Google built a separate shuffle service on top of disaggregated distributed memory. This service backs BigQuery and supports other services, such as [Google Dataflow](https://cloud.google.com/products/dataflow?hl=en).*
>
> **― I spent 4 hours learning the architecture of BigQuery's storage engine - Vu Trinh** [[Link](https://vutr.substack.com/p/i-spent-4-hours-learning-the-architecture)]

> 1. ***Extract**: The process’s first step is extraction. The needed data is gathered from various sources, such as relational databases or third-party APIs*
> 2. ***Transform**: Extracted data undergoes many potential transformations, including cleaning, filtering, combining from different sources, and formatting to conform to a target schema.*
> 3. ***Load**: The transformed data is loaded into the destination with the predefined schema and constrained.*
>
> *ELT solves many of the problems associated with ETL.*
>
> *Most transformation logic can now be handled within the data warehouse using SQL, making it more accessible for users such as data analysts or data scientists. This eliminates the potential performance bottleneck of ETL pipelines.* 
>
> *Most importantly, ELT allows you to keep raw data in the warehouse. This approach offers several advantages. You don’t need to plan transformation logic in advance; instead, the logic can evolve over time based on analytical needs—an especially valuable benefit in today’s agile software development environment.*
>
> **― ETL and ELT - Vu Trinh** [[Link](https://vutr.substack.com/p/etl-and-elt)]

> **Apache Airflow Overview - Vu Trinh** [[Link](https://vutr.substack.com/p/apache-airflow-overview)]

> **How did Airbnb build their semantic layer? - Vu Trinh** [[Link](https://vutr.substack.com/p/how-did-airbnb-build-their-semantic)]

> *In a meltdown, discipline beats brilliance.*
>
> *Over the years, I’ve found that having a simple rule-based system helps me stay grounded. Here are the 4 rules I follow to protect my portfolio:*
>
> - *I invest a fixed amount monthly — rain or shine.*
> - *I don’t add to losers — keeping them relatively small.*
> - *I don’t sell winners — staying the course and being patient.*
> - *I invest for at least 5 years — to give compounding time to work.*
>
> **― Bear Market Survival Guide - App Economy Insights** [[Link](https://www.appeconomyinsights.com/p/bear-market-survival-guide)]

> **Salesforce & AI Strategy - Generative Value** [[Link](https://www.generativevalue.com/p/salesforce-and-ai-strategy)]

This article discusses the history of Salesforce, what made it successful, the state of the business, and the AI opportunity (or threat) today.

> **Everything Wrong with MCP - Shrivu's Substack**  [[Link](https://blog.sshh.io/p/everything-wrong-with-mcp)]

> **How to future-proof your career in the age of AI - Operator's Handbook** [[Link](https://www.operatorshandbook.com/p/how-to-future-proof-your-career-in)]

**Key Takeaways:**

The author’s call to "lean into human strengths while actively engaging with AI" is a compelling middle path. The essay underscores that the future belongs to those who combine AI literacy with irreplaceable human skills—judgment, influence, and adaptability.

**Human Competitive Advantages**:

- **Judgment & Conviction**: Ability to make decisions with incomplete/ambiguous data. Distinguishing impactful work from "interesting but useless" projects. Simplifying complexity into actionable frameworks.
- **Influence & Execution**: Navigating organizational politics and incentives. Building trust and adoption for AI-driven outputs. Understanding unspoken processes and relationships.

**Actionable Skills to Cultivate**:

- Develop "taste" by studying excellence in your field.
- Gain hands-on experience to pressure-test AI outputs.
- Learn to align stakeholders and drive consensus.
- Build strong interpersonal relationships and reputation.

**Adaptability as the Ultimate Skill**:

- AI will keep evolving, so continuous learning and flexibility are critical.
- Focus on areas where humans add unique value (judgment, influence, creativity).

This is a very interesting point: "Develop "taste" by studying excellence in your field."

Just like any skill, taste sharpens with exposure and effort. The more you study, critique, and create, the better you’ll get at recognizing—and producing—excellence. In a world flooded with AI-generated content, the people who thrive will be those who can separate the remarkable from the mediocre.

## Blogs and Articles

> **How Airbnb Standardized Metric Computation at Scale - Airbnb Blog** [[Link](https://medium.com/airbnb-engineering/airbnb-metric-computation-with-minerva-part-2-9afe6695b486)]

> **Digital hygiene - Andrej Karpathy** [[Link](https://karpathy.bearblog.dev/digital-hygiene/)]

Good tips and tricks for digital hygiene, given the pervasive nature of internet fraud and the data collection practices of major tech companies.

> **Measuring AI Ability to Complete Long Tasks - METR** [[Link](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)]

![metr-length-of-tasks-log](2025-April/metr-length-of-tasks-log.png)

> **The "think" tool: Enabling Claude to stop and think in complex tool use situations - Anthropic** [[Link](https://www.anthropic.com/engineering/claude-think-tool)]

Anthropic introduces a "think" tool designed to enhance Claude's complex problem-solving by providing a dedicated space for structured reasoning during tasks. This tool differs from extended thinking by allowing Claude to pause and consider necessary information mid-response, particularly beneficial for multi-step processes and tool use. Evaluations on benchmarks like τ-Bench demonstrated significant performance improvements, especially in policy-heavy domains like airline customer service, where optimized prompting alongside the "think" tool proved most effective. 

> **Tiny Agents: a MCP-powered agent in 50 lines of code - HuggingFace** [[Link](https://huggingface.co/blog/tiny-agents)]

> **Anthropic CEO wants to open the black box of AI models by 2027 - Techcrunch** [[Link](https://techcrunch.com/2025/04/24/anthropic-ceo-wants-to-open-the-black-box-of-ai-models-by-2027/?guccounter=1)]
>
> *Powerful AI will shape humanity’s destiny, and we deserve to understand our own creations before they radically transform our economy, our lives, and our future.*
>
> **― The Urgency of Interpretability - Dario Amodei** [[Link](https://www.darioamodei.com/post/the-urgency-of-interpretability)]

Interpretability isn’t just academic—it’s a prerequisite for safe, controllable AI. The window to solve it is narrowing as AI grows more powerful. By steering resources toward this goal now, we might avoid a future where humanity builds systems it doesn’t understand but can’t afford to stop.

> **The Jobs That Will Fall First As AI Takes Over The Workplace - Forbes** [[Link](https://www.forbes.com/sites/jackkelly/2025/04/25/the-jobs-that-will-fall-first-as-ai-takes-over-the-workplace/)]

**Takeaways:**

1. Timeline for Disruption:
   - By 2030: 30% of U.S. jobs could be automated (McKinsey).
   - By 2035: White-collar restructuring in finance, legal, and media (Larry Fink, Jamie Dimon).
   - By 2045: 50% of jobs may be fully automated (Goldman Sachs).
   - By 2050: AI could dominate 60-80% of jobs, depending on innovation pace.
2. Most Vulnerable Jobs (Near-Term):
   - Administrative: Data entry, scheduling, customer service (60% automatable, per IPPR).
   - Finance & Legal: Bookkeeping, contract drafting, paralegal work (AI tools like Harvey already achieve 90% accuracy).
   - Creative & Media: Basic graphic design, copywriting, journalism (30% at risk by 2035, Pew Research).
   - Routine STEM Tasks: Coding, data analysis (40% automatable by 2040, WEF).
3. More Resilient Jobs (Longer-Term):
   - Healthcare: Nursing, therapy, and patient care (empathy-driven roles).
   - Skilled Trades: Construction, repair, maintenance (physical labor is harder to automate).
   - Education & Leadership: Teaching, high-level management (requires emotional intelligence).

**To protect career:**

- Focus on critical thinking, creativity, and AI collaboration (e.g., prompt engineering, AI-augmented decision-making).
- Target Resilient Sectors- Healthcare, education, skilled trades, and AI-adjacent roles (e.g., cybersecurity, AI ethics).
- Push for employer or government-sponsored programs to transition into hybrid (human + AI) roles.
- Embrace Hybrid Roles- Jobs that combine technical skills with human judgment (e.g., AI-assisted healthcare diagnostics) will thrive.

As Ray Dalio warns, the economy faces a "great deleveraging" where AI disrupts jobs faster than new ones emerge. The key is **adaptability**—those who proactively reinvent their skills today will shape the workforce of tomorrow.

> **Curation is the new leadership superpower. Here are 3 ways to adopt a curation mindset - FastCompany** [[Link](https://www.fastcompany.com/91322752/curation-is-the-new-leadership-superpower-here-are-3-ways-to-adopt-a-curation-mindset)]

The most transformative leaders of the next decade will be those who master the art of curation—seeing their role as a conduit for the best ideas, not the source of them.

The Obsolescence of the "Omniscient Leader": The pace of change, hyper-specialization, and interconnected challenges (e.g., AI, climate, global markets) make it impossible for one person to have all the answers. Leaders must shift from being "the smartest in the room" to becoming "architects of collective intelligence."

**Curation as the Core Leadership Skill:**

1. Curating Talent: Prioritize cognitive diversity over homogeneity. Example: Diverse teams solve problems faster (39% efficiency boost).
2. Curating Ideas: Create systems where unconventional thinking flourishes (e.g., Google’s 20% time → Gmail, Maps). Actively seek "outliers" (contrarians, outsiders) to challenge groupthink.
3. Curating Innovation: Design for "structured serendipity" (e.g., Pixar’s open office, IDEO’s cross-industry brainstorming). Embrace cross-disciplinary collisions (e.g., NASA’s tech inspiring sportswear, biomimicry in architecture).

**How to Cultivate a Curation Mindset**:

- Facilitate, don’t dictate: Ask better questions; let solutions emerge from debate (e.g., Amazon’s "Disagree and Commit").
- Optimize for collaboration, not just efficiency: Space matters (physical or virtual).

> **Perplexity CEO says its browser will track everything users do online to sell ‘hyper personalized’ ads - TechCrunch** [[Link](https://techcrunch.com/2025/04/24/perplexity-ceo-says-its-browser-will-track-everything-users-do-online-to-sell-hyper-personalized-ads/)]

Perplexity is building a browser (Comet) to track user behavior across the web—explicitly to fuel targeted advertising. It highlights the company’s ambition to emulate Google’s surveillance-capitalism playbook. 

Perplexity’s move confirms that the AI search revolution is less about displacing Google’s model than replicating it—with AI as a smarter wrapper for the same ads. 

> **Today’s Most Crucial Leadership Skill Is Systems Thinking - Forbes** [[Link](https://www.forbes.com/sites/maryjohnstone-louis/2025/04/25/todays-most-crucial-leadership-skill-is-systems-thinking/)]

Leaders who master systems thinking don’t just survive uncertainty—they **thrive in it**, turning complexity into competitive advantage.

**Five Key Tools of Systems Thinking for Strategic Leaders**

1. Problem Statements: Move from surface-level fixes to systemic solutions. Example: Instead of asking, *“How do we get customers to recycle?”*, ask, *“How can we redesign products and infrastructure for circularity?”*
2. Stakeholder Mapping: Identify all affected parties—not just obvious ones. Example: For electric vehicles, consider miners of critical minerals, urban planners, and regulators, not just automakers and buyers.
3. Iceberg Analysis: Look beneath visible events to uncover hidden structures and mindsets. Example: Employee burnout isn’t just about workload—it’s shaped by corporate culture, incentive systems, and societal norms.
4. Causal Loops: Visualize feedback loops to see how actions create ripple effects. Example: A cost-cutting measure in one department may increase inefficiencies elsewhere.
5. Iteration & Testing: Embrace adaptive strategies, not rigid plans. Example: Pilot small-scale solutions, measure impact, and refine before full rollout.

> **Perplexity CEO shares the Elon Musk–inspired mantra that helped him build the $9 billion rival to OpenAI - Fortune** [[Link](https://fortune.com/article/perplexity-ceo-aravind-srinivas-elon-musk-inspired-mantra-help-build-billion-dollar-openai-google-rival/)]

Srinivas’s journey highlights resilience, speed, and Silicon Valley’s tight-knit founder network as key drivers of startup success.

1. "It’s Only Over When You Give Up" – Aravind Srinivas, CEO of AI search startup Perplexity, draws inspiration from Elon Musk’s perseverance during SpaceX’s early failures. He told Harvard students that success comes from relentless self-belief, even when others doubt you.
2. Rocketing Valuation – Perplexity, competing with Google and OpenAI, grew from a 1B to 9B and is now in talks to raise funds at an 18B valuation.
3. Forget Pitch Decks, Build Fast – Srinivas advises founders to focus on rapid product iteration rather than lengthy business plans. He admits he doesn’t even know how to make a pitch deck—Perplexity’s success came from live demos.
4. OpenAI Alumni Network – Despite competing with OpenAI, Srinivas maintains a strong relationship with Sam Altman (his former boss at OpenAI). This mirrors the "PayPal Mafia" dynamic, where ex-OpenAI employees now lead major AI firms like Anthropic and Safe Superintelligence.

> **Marc Andreessen predicts one of the few jobs that may survive the rise of AI automation - Fortune** [[Link](https://fortune.com/article/mark-andreessen-venture-capitalism-ai-automation-a16z/)]

Andreessen’s logic suggests focusing on roles where trust, psychology, and networks matter more than data crunching. But don’t underestimate AI’s ability to creep into those domains too.

> **How To Get Noticed Without Self-Promotion By Using Strategic Visibility - Forbes** [[Link](https://www.forbes.com/sites/cynthiayoung/2025/04/27/how-to-get-noticed-without-self-promotion-by-using-strategic-visibility/)]

**Core Lessons:**

1. **Hard Work ≠ Visibility**: Doing great work is necessary but insufficient. If leaders don’t know what you’re doing, they can’t reward it. Waiting for annual reviews is too late—visibility requires consistent, intentional updates.
2. **Humility Has a Hidden Cost**: While modesty is admirable, staying silent can render you invisible. Gallup’s data on declining engagement (just 36% in 2020) highlights how disengagement hurts promotion prospects. Visibility isn’t ego-driven; it’s about ensuring your impact is recognized.
3. **Visibility ≠ Bragging**: Framing contributions as useful knowledge (e.g., "Here’s how I solved X") builds trust and leadership credibility. Sharing wins, failures, and best practices helps the team and positions you as a problem-solver.
4. **Tactical Ways to Increase Visibility**
   - Share knowledge: Lead "lessons learned" sessions or contribute to internal newsletters.
   - Mentor others: Their success reflects your leadership.
   - Speak up strategically: One substantive insight per meeting > empty chatter.
   - Volunteer for high-impact projects: Align with organizational priorities.
   - Write internally: Document best practices to showcase thought leadership.
5. **Emotional Intelligence (EQ) Matters More Than Extroversion**
   - Visibility is about meaningful engagement, not being the loudest.
   - Avoid self-deprecating language ("I’m sorry, but…")—speak with conviction.
6. **What Leaders Actually Notice**
   - Initiative, influence, and alignment with goals matter more than face-time.
   - Working smart (not just late) and collaborating effectively signal leadership potential.

## YouTube and Podcast

> **DOGE updates + Liberation Day Tariff Reactions with Ben Shapiro and Antonio Gracias - All-In Podcast** [[Link](https://www.youtube.com/watch?v=OjhA9p3ZXW0&ab_channel=All-InPodcast)]

> **2027 Intelligence Explosion: Month-by-Month Model — Scott Alexander & Daniel Kokotajlo - Dwarkesh Patel** [[Link](https://www.youtube.com/watch?v=htOvH12T7mU&ab_channel=DwarkeshPatel)]

> **Trump vs Harvard, Nvidia export controls, how DEI killed Hollywood with Tim Dillon - All-In Podcast** [[Link](https://www.youtube.com/watch?v=rCrb4TbHRxc&ab_channel=All-InPodcast)]

> **E187 | 关税战难解美国制造业困境，旧秩序正在崩溃 - 硅谷101播客** [[Link](https://www.youtube.com/watch?v=6Fmo_jBc7mA&t=1637s&ab_channel=%E7%A1%85%E8%B0%B7101%E6%92%AD%E5%AE%A2)]

> **How DeepSeek Rewrote the Transformer [MLA] - Welch Labs** [[Link](https://www.youtube.com/watch?v=0VLAoVGf_74&ab_channel=WelchLabs)]

A lecture explaining the architecture and optimizations behind DeepSeek R1, a language model that improves Transformer efficiency.

> **Live Demo: Reinforcement Fine-Tuning for LLMs — Build Smarter Models with Less Data l Tutorial - Predibase** [[Link](https://www.youtube.com/watch?v=tHvZypO3le4&ab_channel=Predibase)]

This video was talking about why RFT beats supervised fine-tuning (SFT) in reasoning tasks, giving live demo of an end-to-end RFT workflow, and PyTorch-to-Triton case study showing real-world impact.

> **Model Context Protocol (MCP), clearly explained (why it matters) - Greg Isenberg** [[Link](https://www.youtube.com/watch?v=7j_NE6Pjv-E&ab_channel=GregIsenberg)]

> **Trump Rally or Bessent Put? Elon Back at Tesla, Google's Gemini Problem, China's Thorium Discovery - All-In Podcast** [[Link](https://www.youtube.com/watch?v=4tcd1EgQrhU&ab_channel=All-InPodcast)]

> *Suffering is mostly mental anguish and mental pain and it just means you don't want to do the task at hand.* 
>
> *The kind of fame that pure actors and celebrities have, I wouldn't want, but the kind of fame that's earned because you did something useful, why dodge that.*
>
> *People will always want more status uh but I think you can be satisfied at a certain level of wealth.*
>
> *Not the kind of confidence that would say I have the answer but the kind of confidence that I will figure it out and I know what I want or only I am a good arbiter of what I want.*
>
> *Pride is the enemy of learning, so when I look at my friends and colleagues, the ones who are still stuck in the past and have grown the least are the ones who were the proudest, because they sort of feel like they already had the answers and so they don't want to correct themselves publicly.*
>
> *I think everybody puts themselves first that's just human nature, you're here because you survive you're a separate organism.*
>
> *The happier you are, the more you can sustain doing something, the more likely you're going to do something that will in turn make you even happier, and you'll continue to do it, and you'll outwork everybody else. The more free you are the better you can allocate your time.* 
>
> *There are no problems in the real world other than maybe things that inflict pain on your body. Everything else has to become a problem in your mind first.*
>
> *Your family is broken but you're going to fix the world. People are running out there to try and fix the world when their own lives are a mess.*
>
> *I think the only true test of intelligence is if you get what you want out of life and there are two parts to that one is getting what you want so you know how to get it and the second is wanting the right things knowing what to want in the first place.*
>
> *Usually I think people end up there because they are going on autopilot with sort of societal expectations or other people's expectations or out of guilt or out of like mimetic desire.*
>
> *Probably the biggest regret will be staying in the relationship after you knew it was over, exactly you should have left sooner, the moment you knew it wasn't going to work out, you should have moved on.*
>
> *We are naturally hardwired to be pessimists but modern society is very different despite whatever problems you may have with modern society, it is far far safer than living in the jungle and just trying to survive and the opportunities.*
>
> *Leave all those labels alone. It's better just to look at the problem at hand, look at reality the way it is, try to take yourself out of the equation in a sense.*
>
> *The less you think about yourself the more you can think about a mission or about God or about a child or something like that.*
>
> *I don't think there are any formulas i think it's unique to each person it's like asking a successful person how did you become successful each one of them will give you a different story uh you can't follow anyone else's path.*
>
> *A lot of change is more about desire and understanding than it is about uh forcing yourself or trying to domesticate yourself.*
>
> *When your mind is under stress, it's because it has two conflicting desires at once... and anxiety I think is sort of this pervasive unidentifiable stress where you're just kind of stressed out all the time and you're not even sure why and you can't even identify the underlying problem. I think the reason for that is because you have so many unresolved problems unresolved stress points that have piled up in your life that you can no longer identify what the problems are.*
>
> *Life is going to play out the way it's going to play out there will be some good and some bad most of it is actually just up to your interpretation.*
>
> *The gut is what decides the head, is kind of what rationalizes it afterwards, the gut is the ultimate decision maker.*
>
> *You can't change other people, you can change your reaction to them.*
>
> *If you do want to change someone's behavior, I I think the only effective way to do it is to compliment them when they do something you want, not to insult them or be negative or critical when they do something you don't want.*
>
> If you can't decide, the answer is no.
>
> *Almost invariably the advice that you would give yourself 10 years ago is still the advice that you need to hear today.*
>
> *On mental things, I think understanding is way more important once you see the truth of something you cannot unsee it... when we really do see something clearly, it changes our behavior immediately, and that is far more efficient than trying to change your behavior through repetition.*
>
> *Truth is often painful, if it wasn't, we'd all be seeing truth all the time. Reality is always reflecting truth that's all it is why would you not have accessed it already exactly... wisdom is the set of things that cannot be transmitted. If they could be transmitted you know we'd read the same five philosophy books, and we'd all be done, we'd all be wise. You have to learn it for yourself, it has to be rediscovered for yourself in your own context.*
>
> *You're probably better off only caring about things that are local or things that you can affect. So if you really care about something that's in the news, then by all means care about it but make a difference go do something about it.*
>
> *Desire is a contract to be unhappy until you get what you want.*
>
> *The real currency of life is attention it's what you choose to pay attention to and and what you do about it.*
>
> **― 44 Harsh Truths About Human Nature - Naval Ravikant (4K) - Chris Williamson** [[Link](https://www.youtube.com/watch?v=KyfUysrNaco)]

Key Learnings:

* Someone who can do the job peacefully or happily is more effective than someone with unnecessary emotional turmoil
* Fame sought for its own sake is fragile and leads to a constant need to perform.
* People often say things they don't really believe, driven by a desire to be seen as something they are not.
* Status is zero-sum and insatiable, unlike wealth. Status is often comparative, like leaderboards, where one person's gain can be another's loss.
* Self-esteem comes from aligning actions with internal values, especially when difficult. Genuine sacrifice, doing something you want less for something you value more, can build self-esteem.
* True confidence is not having all the answers but the self-belief to figure things out. 
* Pride is an enemy of learning and can lead to being stuck in past mistakes. 
* Everyone puts themselves first; unapologetic self-prioritization is rare but perhaps more honest. Much of what appears as altruism might be a waste of time if it goes against one's true desires.
* Happiness and freedom are intertwined with efficiency and productivity. 
* Many emotional problems arise from the mind creating problems where none exist in the real world. He advises observing one's thoughts objectively to realize unnecessary emotional energy expenditure.
* People often try to fix the world while their own lives are in disarray. He questions the credibility of those who cannot manage their own lives but seek to solve global issues.
* True intelligence is getting what you want out of life by wanting the right things and knowing how to get them. 
* Many people go through life unconsciously following societal or mimetic desires. He emphasizes the importance of thinking things through for oneself rather than blindly following others.
* Staying too long in bad situations (relationships, jobs) is a common regret. 
* We are naturally hardwired for pessimism due to evolutionary pressures to avoid ruin. 
* Humans are dynamic and labels like optimist, pessimist, introvert, extrovert are self-limiting.
* Overthinking about oneself can lead to misery; focusing on something bigger can bring happiness. Overthinking and rumination do not help with happiness.
* There are no universal formulas for success or happiness; each person's path is unique. 
* Lasting change comes from desire and understanding, not forcing oneself. He suggests aligning actions with genuine wants for maximal effectiveness.
* Anxiety often stems from having many unresolved and conflicting desires. 
* Our interpretations of experiences shape our reality. The same experience can lead to different emotional responses based on individual interpretation.
* The "gut" is the ultimate decision-maker, representing refined judgment accumulated through evolution and experience. He advises trusting this instinct once it's developed.
* You cannot change other people, only your reaction to them. He adds that people change through their own insights or trauma, not by being told to.
* Negative reinforcement is less effective than positive reinforcement in changing behavior. 
* If faced with a difficult choice and unable to decide, the answer is often "no." He also suggests that when choosing between two equal options, take the more painful path in the short term.
* Understanding is more important than discipline for mental change.
* Truth, though often painful, is constantly reflected by reality; wisdom is the personal rediscovery and contextual application of timeless truths. He also mentions that many important life lessons are "unteachable" in the sense that they must be experienced firsthand to be truly understood.
* Memorization is becoming less valuable in the age of readily available information; understanding, judgment, and taste are more crucial. He links understanding to solving real problems and finding generalizable truths.
* Philosophy evolves with new knowledge and perspectives. He explains how advancements in science and technology lead to different philosophical outlooks, and even moral philosophy progresses over time.
* Many philosophical paradoxes can be resolved by considering different scales and timeframes. Naval suggests that seemingly contradictory questions like free will and determinism can be understood by shifting perspectives.
* Coordination is essential for societal function; pure libertarianism is unsustainable. 
* Modern AI, while powerful, currently lacks true creativity and deep understanding. 
* Meaning can be more important than moment-to-moment happiness.
* In an age of news saturation, it's a battle to maintain focus on what truly matters and what one can influence. He emphasizes that attention is the real currency of life and should be spent consciously. Attention, not time or money, is the most fundamental resource in life. 
* Getting past one's past is a skill achieved by processing it to be rid of it, not to dwell on it. 

> *I think agents are real, but I think that we are far away from that because we're still at the phase of how do you build reliable software in production for an enterprise versus the toy apps that you see on the internet which is like let me vibe code something. I think these things are worlds apart still. - Chamath Palihapitiya*
>
> *I think we have not yet figured out how to move the budgets from experimentation to mainline production. Meaning where large chunks of the US economy are comfortable enough with the ways in which hallucinations are managed such that they will replace legacy deterministic code with this new probabilistic model generated code meaning model enabled code. - Chamath Palihapitiya*
>
> **― Trump's First 100 Days, Tariffs Impact Trade, AI Agents, Amazon Backs Down - All-In Podcast** [[Link](https://www.youtube.com/watch?v=W960TW79QCI)]

## Papers and Reports

> **Orchestrating Agents and Data for Enterprise: A Blueprint Architecture for Compound AI** [[Link](https://arxiv.org/abs/2504.08148)]

This paper contributes to the enterprise AI landscape by offering a comprehensive architectural blueprint for deploying agentic, modular, and data-integrated AI systems that can efficiently leverage LLMs and enterprise assets. 

## Github

> **Google Gemini 2.0 with MCP (Model Context Protocol) Servers - Gemini Samples** [[Link](https://github.com/philschmid/gemini-samples/blob/main/examples/gemini-mcp-example.ipynb)]

> **Maestro - A Framework for Claude Opus, GPT and local LLMs to Orchestrate Subagents - maestro** [[Link](https://github.com/Doriandarko/maestro)]

> **MCP-Agent** [[Link](https://github.com/lastmile-ai/mcp-agent)]

> **Local Deep Researcher** [[Link](https://github.com/langchain-ai/local-deep-researcher)]

## News

> **Accelerate Generalist Humanoid Robot Development with NVIDIA Isaac GR00T N1 - NVIDIA** [[Link](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/)]

> **Announcing the Agent2Agent Protocol (A2A) - Google for Developers** [[Link](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)]

**Key Takeaways:**

1. A2A is an open-source protocol backed by 50+ tech giants (e.g., Salesforce, SAP, Cohere) and consultancies (e.g., Accenture, Deloitte). It allows agents from different vendors/frameworks to communicate, share data, and coordinate tasks without being locked into a single platform.
2. Solving Enterprise Pain Points: Breaks down silos by letting agents interoperate across HR (Workday), CRM (Salesforce), ERP (SAP), and other systems. Example: A hiring manager’s agent can autonomously source candidates, schedule interviews, and run background checks by collaborating with specialized agents.
3. Design Principles:
   - Agent-Centric: Supports unstructured, multi-agent collaboration (beyond rigid "tool" roles).
   - Built on Standards: Uses HTTP, JSON-RPC, and SSE for easy integration.
   - Secure by Default: Enterprise-grade auth (aligned with OpenAPI).
   - Long-Running Tasks: Handles tasks lasting hours/days with real-time updates.
   - Multimodal: Supports text, audio, video, and UI negotiations (e.g., web forms, iframes).

> **The Chinese goods Americans most rely on, from microwaves to Barbies - Financial Times** [[link](https://www.ft.com/content/ec96e2ed-5dd6-4c6b-92a0-1b77bf517b36)]

> **Apple Vision Pro 2 Reportedly Cheaper & Lighter, Mac-Tethered Headset Coming Too - Upload** [[Link](https://www.uploadvr.com/apple-vision-pro-2-reportedly-cheaper-lighter-mac-tethered-headset-coming-too/)]

## Others

Machine learning (ML) solutions applied to business problems across various industries:

* ML and LLM system design: 500 case studies to learn from [[Link](https://www.evidentlyai.com/ml-system-design)]
* Machine Learning and Data Science Applications in Industry - Firmai [[Link](https://github.com/firmai/industry-machine-learning)]
* Business Machine Learning - Firmai [[Link](https://github.com/firmai/business-machine-learning)]
* ML System Design Case Studies Repository [[Link](https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies)] 
* 500+ Artificial Intelligence Project List with Code [[Link](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code)]

