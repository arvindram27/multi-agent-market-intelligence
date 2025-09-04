import Image from "next/image";

// Component for individual agent spotlight
const AgentSpotlight = ({ iconSrc, title, text }: { iconSrc: string, title: string, text: string }) => (
  <div className="bg-gray-800 bg-opacity-50 p-6 rounded-lg shadow-xl hover:shadow-cyan-500/50 transition-shadow duration-300 backdrop-blur-sm">
    <div className="flex items-center mb-4">
      <Image src={iconSrc} alt={`${title} icon`} width={48} height={48} className="mr-4 rounded-md" />
      <h3 className="text-xl font-semibold text-cyan-400">{title}</h3>
    </div>
    <p className="text-gray-300 text-sm">{text}</p>
  </div>
);

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans">
      {/* Hero Section */}
      <section 
        className="relative h-screen flex flex-col items-center justify-center text-center p-8 bg-cover bg-center"
        style={{ backgroundImage: "url(\"/assets/hero_image_symbiotic_intelligence.png\")" }}
      >
        <div className="absolute inset-0 bg-black opacity-60"></div> {/* Overlay for text readability */}
        <div className="relative z-10">
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            Market Intelligence. <span className="text-cyan-400">Evolved.</span>
          </h1>
          <p className="text-xl md:text-2xl text-gray-300 mb-10 max-w-3xl mx-auto">
            Experience the synergy of AI agents working in concert to transform raw data into your decisive competitive advantage.
          </p>
          <a 
            href="#ecosystem-section" 
            className="bg-cyan-500 hover:bg-cyan-400 text-white font-bold py-3 px-8 rounded-lg text-lg transition duration-300 shadow-lg hover:shadow-cyan-500/50 transform hover:scale-105"
          >
            Unveil the Future
          </a>
        </div>
      </section>

      {/* The Problem / The Opportunity Section */}
      <section className="py-20 px-8 bg-gray-800 bg-opacity-30">
        <div className="container mx-auto max-w-4xl text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 text-cyan-300">Navigating the Data Deluge: The Imperative for Intelligent Evolution</h2>
          <p className="text-lg text-gray-300 mb-4">
            In today&apos;s hyper-connected world, businesses are drowning in data yet starving for wisdom. Traditional market intelligence struggles to keep pace with the sheer volume, velocity, and variety of information. Opportunities are missed, threats emerge unseen, and decisions are made on incomplete pictures.
          </p>
          <p className="text-lg text-gray-300">
            But what if you could transcend these limitations? What if you had a tireless, intelligent partner sifting through the noise, identifying critical signals, and delivering actionable insights 24/7? The convergence of advanced AI and agent-based systems presents an unprecedented opportunity to transform market intelligence from a reactive chore into a proactive, strategic powerhouse.
          </p>
        </div>
      </section>

      {/* Introducing the AI Agent Ecosystem Section */}
      <section id="ecosystem-section" className="py-20 px-8 bg-gray-900">
        <div className="container mx-auto max-w-5xl text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 text-cyan-300">Symbiotic Intelligence: Your Dedicated AI Agent Ecosystem</h2>
          <p className="text-lg text-gray-300 mb-12 max-w-3xl mx-auto">
            Symbiotic Intelligence is not just another tool; it&apos;s a dynamic ecosystem of specialized AI agents, each an expert in its domain, working in concert to provide you with a comprehensive, multi-faceted understanding of your market landscape. Imagine a dedicated team of AI analysts, always learning, always vigilant.
          </p>
          <div className="grid md:grid-cols-2 gap-8">
            <AgentSpotlight 
              iconSrc="/assets/icon_news_agent.png"
              title="Real-Time Insight Engine"
              text="Our News Agent scans global sources, identifies relevant developments, and delivers concise, actionable summaries, keeping you perpetually informed."
            />
            <AgentSpotlight 
              iconSrc="/assets/icon_competitor_agent.png"
              title="Strategic Foresight"
              text="Uncover competitor strategies, product launches, and market positioning with unparalleled depth. Turn their moves into your advantage."
            />
            <AgentSpotlight 
              iconSrc="/assets/icon_trend_agent.png"
              title="Horizon Scanner"
              text="Identify nascent market shifts, emerging technologies, and evolving consumer behaviors before they become mainstream. Innovate with confidence."
            />
            <AgentSpotlight 
              iconSrc="/assets/icon_sentiment_agent.png"
              title="Audience Pulse"
              text="Go beyond demographics. Our Sentiment Agent analyzes real-time public discourse to reveal the true feelings, motivations, and pain points of your target audience."
            />
          </div>
        </div>
      </section>

      {/* Placeholder for The Impact / Benefits Section */}
      <section className="py-20 px-8 bg-gray-800 bg-opacity-30">
        <div className="container mx-auto max-w-4xl text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 text-cyan-300">The Symbiotic Advantage: Transform Insight into Impact</h2>
          <p className="text-lg text-gray-300">Content for Users, Investors, and Talent will go here, highlighting benefits and including CTAs like "See It In Action", "Explore Investment Profile", and "Join Our Quest".</p>
        </div>
      </section>

      {/* Placeholder for The Vision / The Future Section */}
      <section className="py-20 px-8 bg-gray-900">
        <div className="container mx-auto max-w-4xl text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 text-cyan-300">The Horizon of Intelligence: Ever Evolving, Ever Learning</h2>
          <p className="text-lg text-gray-300">This section will paint a picture of the long-term vision, discussing continuous evolution and the future of synergistic human-AI market understanding.</p>
        </div>
      </section>

      {/* Placeholder for Call to Action / Engagement Section */}
      <section className="py-20 px-8 bg-gray-800 bg-opacity-30">
        <div className="container mx-auto max-w-4xl text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 text-cyan-300">Connect with Symbiotic Intelligence</h2>
          <p className="text-lg text-gray-300">This section will feature primary CTAs for Users (Request Early Access/Demo), Investors (Inquiries), and Talent (Careers), along with an optional newsletter signup.</p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 px-8 text-center text-gray-500 border-t border-gray-700">
        <p>&copy; {new Date().getFullYear()} Symbiotic Intelligence. All Rights Reserved.</p>
        <div className="mt-2">
          <a href="#" className="hover:text-cyan-400 mx-2">Privacy Policy</a>
          <a href="#" className="hover:text-cyan-400 mx-2">Terms of Service</a>
          {/* Add social media links/icons here */}
        </div>
      </footer>
    </div>
  );
}

