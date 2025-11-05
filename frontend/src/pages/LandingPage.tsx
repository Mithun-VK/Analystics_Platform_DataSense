// src/pages/LandingPage.tsx

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowRight,
  Check,
  Zap,
  BarChart3,
  Database,
  Shield,
  TrendingUp,
  Sparkles,
  ChevronRight,
  Play,
  Star,
  Mail,
  Github,
  Linkedin,
  Twitter,
} from 'lucide-react';
import Button from '@/components/shared/Button';
import { useAuth } from '@/hooks/useAuth';

interface Feature {
  icon: React.ElementType;
  title: string;
  description: string;
}

interface Testimonial {
  name: string;
  role: string;
  company: string;
  content: string;
  initials: string;
  rating: number;
}

interface PricingPlan {
  name: string;
  price: string;
  period?: string;
  description: string;
  features: string[];
  cta: string;
  highlighted: boolean;
}

/**
 * LandingPage - Professional landing page with modern design
 * Features: Hero, Features, Pricing, Testimonials, FAQ, CTA, Footer
 */
const LandingPage = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [expandedFAQ, setExpandedFAQ] = useState<number | null>(null);
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Redirect if logged in
  useEffect(() => {
    if (user) navigate('/dashboard');
  }, [user, navigate]);

  const features: Feature[] = [
    {
      icon: BarChart3,
      title: 'Advanced Analytics',
      description: 'Statistical analysis and data exploration at scale',
    },
    {
      icon: Sparkles,
      title: 'AI-Powered Insights',
      description: 'Machine learning-driven actionable intelligence',
    },
    {
      icon: TrendingUp,
      title: 'Custom Visualizations',
      description: 'Beautiful, interactive charts and dashboards',
    },
    {
      icon: Database,
      title: 'Data Management',
      description: 'Organize, clean, and transform datasets efficiently',
    },
    {
      icon: Shield,
      title: 'Enterprise Security',
      description: 'End-to-end encryption and compliance certified',
    },
    {
      icon: Zap,
      title: 'Lightning Performance',
      description: 'Process massive datasets in seconds',
    },
  ];

  const testimonials: Testimonial[] = [
    {
      name: 'Sarah Johnson',
      role: 'Data Analyst',
      company: 'TechCorp',
      content:
        'Transformed how we analyze data. The AI insights are incredibly accurate and save hours weekly.',
      initials: 'SJ',
      rating: 5,
    },
    {
      name: 'Michael Chen',
      role: 'CEO',
      company: 'StartupXYZ',
      content:
        'From days of analysis to minutes of insights. Best investment for our analytics team.',
      initials: 'MC',
      rating: 5,
    },
    {
      name: 'Emily Rodriguez',
      role: 'Product Manager',
      company: 'Enterprise Co',
      content:
        'Intuitive tools with powerful capabilities. Stakeholders love the interactive dashboards.',
      initials: 'ER',
      rating: 5,
    },
  ];

  const pricingPlans: PricingPlan[] = [
    {
      name: 'Starter',
      price: 'Free',
      description: 'Perfect for getting started',
      features: [
        'Up to 5 datasets',
        'Basic analytics',
        '1 GB storage',
        'Community support',
      ],
      cta: 'Get Started',
      highlighted: false,
    },
    {
      name: 'Professional',
      price: '$29',
      period: '/month',
      description: 'For serious data work',
      features: [
        'Unlimited datasets',
        'Advanced analytics',
        'AI-powered insights',
        '100 GB storage',
        'Priority support',
        'Custom visualizations',
      ],
      cta: 'Start Free Trial',
      highlighted: true,
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      description: 'For large organizations',
      features: [
        'Everything in Pro',
        'Unlimited storage',
        'API access',
        'Custom integration',
        'Dedicated support',
        'SLA guarantee',
      ],
      cta: 'Contact Sales',
      highlighted: false,
    },
  ];

  const faqs = [
    {
      question: 'What data formats are supported?',
      answer: 'CSV, Excel (XLS, XLSX), and JSON files up to 100MB each.',
    },
    {
      question: 'Is my data secure?',
      answer:
        'Enterprise-grade encryption with GDPR, HIPAA, and SOC 2 compliance.',
    },
    {
      question: 'Can I export results?',
      answer: 'PNG/SVG charts, CSV data, and PDF reports included.',
    },
    {
      question: 'Is API access available?',
      answer: 'Yes, on Pro and Enterprise plans for custom integrations.',
    },
    {
      question: 'What is your uptime?',
      answer: '99.9% uptime with 24/7 monitoring and support.',
    },
    {
      question: 'Do you offer a trial?',
      answer: '14-day free trial with up to 5 datasets on any plan.',
    },
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setEmail('');
      // Success toast here
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white">
      {/* Navigation */}
      <nav className="sticky top-0 z-40 bg-white border-b border-gray-200">
        <div className="container flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-lg text-gray-900">DataAnalytics</span>
          </div>

          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-gray-600 hover:text-gray-900 transition">
              Features
            </a>
            <a href="#pricing" className="text-gray-600 hover:text-gray-900 transition">
              Pricing
            </a>
            <a href="#testimonials" className="text-gray-600 hover:text-gray-900 transition">
              Testimonials
            </a>
            <a href="#faq" className="text-gray-600 hover:text-gray-900 transition">
              FAQ
            </a>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate('/login')}
            >
              Login
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={() => navigate('/register')}
            >
              Sign Up
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20 md:py-28">
        <div className="container">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-50 border border-blue-200 rounded-full mb-6">
              <Sparkles className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-600">
                AI-powered insights now available
              </span>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
              Transform Your Data Into{' '}
              <span className="bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
                Actionable Insights
              </span>
            </h1>

            <p className="text-xl text-gray-600 mb-8 leading-relaxed max-w-2xl">
              Powerful analytics platform to explore, analyze, and visualize your data with ease.
              Make smarter decisions backed by data.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 mb-12">
              <Button
                variant="primary"
                size="lg"
                leftIcon={ArrowRight}
                onClick={() => navigate('/register')}
              >
                Start Free Trial
              </Button>
              <Button variant="outline" size="lg" leftIcon={Play}>
                Watch Demo
              </Button>
            </div>

            <div className="grid grid-cols-3 gap-8">
              <div>
                <p className="text-3xl font-bold text-gray-900">10K+</p>
                <p className="text-sm text-gray-600 mt-1">Active Users</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-gray-900">50M+</p>
                <p className="text-sm text-gray-600 mt-1">Analyses Run</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-gray-900">99.9%</p>
                <p className="text-sm text-gray-600 mt-1">Uptime</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 md:py-28 bg-gray-50">
        <div className="container">
          <div className="max-w-3xl mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Powerful Features
            </h2>
            <p className="text-lg text-gray-600">
              Everything you need for comprehensive data analysis
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <div
                  key={feature.title}
                  className="card hover:shadow-lg transition-shadow"
                >
                  <div className="card-body">
                    <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                      <Icon className="w-6 h-6 text-blue-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600 text-sm leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-20 md:py-28">
        <div className="container">
          <div className="max-w-3xl mx-auto text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Simple, Transparent Pricing
            </h2>
            <p className="text-lg text-gray-600">
              Choose the perfect plan for your team
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {pricingPlans.map((plan) => (
              <div
                key={plan.name}
                className={`card transition-all ${
                  plan.highlighted
                    ? 'ring-2 ring-blue-600 shadow-xl md:scale-105'
                    : ''
                }`}
              >
                <div className="card-body">
                  {plan.highlighted && (
                    <div className="inline-block bg-blue-600 text-white text-xs font-bold px-3 py-1 rounded-full mb-4">
                      Most Popular
                    </div>
                  )}

                  <h3 className="text-2xl font-bold text-gray-900 mb-2">
                    {plan.name}
                  </h3>
                  <p className="text-gray-600 text-sm mb-6">{plan.description}</p>

                  <div className="mb-6">
                    <span className="text-4xl font-bold text-gray-900">
                      {plan.price}
                    </span>
                    {plan.period && (
                      <span className="text-gray-600 text-sm ml-2">
                        {plan.period}
                      </span>
                    )}
                  </div>

                  <Button
                    variant={plan.highlighted ? 'primary' : 'outline'}
                    fullWidth
                    onClick={() => navigate('/register')}
                    className="mb-8"
                  >
                    {plan.cta}
                  </Button>

                  <div className="space-y-3 border-t border-gray-200 pt-6">
                    {plan.features.map((feature) => (
                      <div key={feature} className="flex items-center gap-3">
                        <Check className="w-5 h-5 text-green-600 flex-shrink-0" />
                        <span className="text-sm text-gray-600">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="py-20 md:py-28 bg-gray-50">
        <div className="container">
          <div className="max-w-3xl mx-auto text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Loved by Teams Worldwide
            </h2>
            <p className="text-lg text-gray-600">
              See why thousands of teams trust DataAnalytics
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial) => (
              <div key={testimonial.name} className="card">
                <div className="card-body">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="font-semibold text-blue-700">
                        {testimonial.initials}
                      </span>
                    </div>
                    <div>
                      <p className="font-semibold text-gray-900">
                        {testimonial.name}
                      </p>
                      <p className="text-sm text-gray-600">
                        {testimonial.role} at {testimonial.company}
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-1 mb-4">
                    {Array.from({ length: testimonial.rating }).map((_, i) => (
                      <Star
                        key={i}
                        className="w-4 h-4 fill-yellow-400 text-yellow-400"
                      />
                    ))}
                  </div>

                  <p className="text-gray-600 leading-relaxed">
                    "{testimonial.content}"
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" className="py-20 md:py-28">
        <div className="container max-w-2xl">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Frequently Asked Questions
            </h2>
            <p className="text-lg text-gray-600">
              Get answers to common questions
            </p>
          </div>

          <div className="space-y-3">
            {faqs.map((faq, index) => (
              <div key={index} className="border border-gray-200 rounded-lg overflow-hidden">
                <button
                  onClick={() =>
                    setExpandedFAQ(expandedFAQ === index ? null : index)
                  }
                  className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition"
                >
                  <span className="font-medium text-gray-900 text-left">
                    {faq.question}
                  </span>
                  <ChevronRight
                    className={`w-5 h-5 text-gray-400 transition-transform ${
                      expandedFAQ === index ? 'rotate-90' : ''
                    }`}
                  />
                </button>

                {expandedFAQ === index && (
                  <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
                    <p className="text-gray-600 leading-relaxed">{faq.answer}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 md:py-28 bg-gradient-to-br from-blue-600 to-blue-700">
        <div className="container">
          <div className="max-w-2xl mx-auto text-center">
            <h2 className="text-4xl font-bold text-white mb-4">
              Ready to Get Started?
            </h2>
            <p className="text-lg text-blue-100 mb-8">
              Join thousands of teams making smarter decisions with data
            </p>

            <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3 max-w-md mx-auto">
              <input
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="flex-1 px-4 py-3 rounded-lg border-0 focus:ring-2 focus:ring-blue-300"
                required
              />
              <Button
                type="submit"
                variant="primary"
                loading={isLoading}
                leftIcon={Mail}
              >
                Get Started
              </Button>
            </form>

            <p className="text-sm text-blue-100 mt-4">
              No credit card required. Free for 14 days.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-400 py-12 md:py-16">
        <div className="container">
          <div className="grid md:grid-cols-4 gap-8 mb-12">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="w-5 h-5 text-blue-400" />
                <span className="font-semibold text-white">DataAnalytics</span>
              </div>
              <p className="text-sm text-gray-500">
                Transform data into actionable insights.
              </p>
            </div>

            {[
              {
                title: 'Product',
                links: ['Features', 'Pricing', 'Security', 'Roadmap'],
              },
              {
                title: 'Company',
                links: ['About', 'Blog', 'Careers', 'Contact'],
              },
              {
                title: 'Legal',
                links: ['Privacy', 'Terms', 'Cookies'],
              },
            ].map((section) => (
              <div key={section.title}>
                <h4 className="font-semibold text-white mb-4">{section.title}</h4>
                <ul className="space-y-2">
                  {section.links.map((link) => (
                    <li key={link}>
                      <a href="#" className="hover:text-white transition">
                        {link}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          <div className="border-t border-gray-800 pt-8">
            <div className="flex flex-col md:flex-row items-center justify-between">
              <p className="text-sm text-gray-500">
                © 2025 DataAnalytics. All rights reserved.
              </p>
              <div className="flex gap-4 mt-4 md:mt-0">
                <a href="#" className="text-gray-400 hover:text-white transition">
                  <Github className="w-5 h-5" />
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition">
                  <Twitter className="w-5 h-5" />
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition">
                  <Linkedin className="w-5 h-5" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
