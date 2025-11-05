// src/pages/RegisterPage.tsx - PREMIUM PRODUCTION GRADE

import { useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  BarChart3,
  Shield,
  CheckCircle,
  ArrowRight,

} from 'lucide-react';
import RegisterForm from '@/components/auth/RegisterForm';
import { useAuth } from '@/hooks/useAuth';

/**
 * RegisterPage - Premium registration page with dark theme
 * Features: Two-column layout, animations, social proof, professional design
 */
const RegisterPage = () => {
  const navigate = useNavigate();
  const { user, isLoading } = useAuth();

  // Redirect if already logged in
  useEffect(() => {
    if (user) {
      navigate('/dashboard');
    }
  }, [user, navigate]);

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin">
            <div className="h-12 w-12 border-4 border-blue-500 border-t-blue-600 rounded-full" />
          </div>
          <p className="text-slate-400 text-sm">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden">
      {/* ====================================================================
          Left Section - Hero (Desktop Only)
          ==================================================================== */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12 relative overflow-hidden">
        {/* Animated background blobs */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-20 left-20 w-64 h-64 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl animate-blob" />
          <div className="absolute top-40 right-20 w-64 h-64 bg-green-500 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-2000" />
          <div className="absolute -bottom-8 left-1/2 w-64 h-64 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-4000" />
        </div>

        {/* Content */}
        <div className="relative z-10">
          {/* Logo */}
          <div className="flex items-center gap-3 mb-12">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-white">DataAnalytics</span>
          </div>

          {/* Hero Content */}
          <h1 className="text-5xl font-bold text-white mb-6 leading-tight">
            Start Your Data Journey Today
          </h1>
          <p className="text-lg text-slate-300 mb-10 leading-relaxed">
            Join thousands of teams transforming data into actionable insights. 
            Get instant access with no credit card required.
          </p>

          {/* Feature List */}
          <div className="space-y-5 mb-12">
            {[
              {
                icon: '⚡',
                title: 'Instant Access',
                desc: 'Start exploring data immediately',
              },
              {
                icon: '🎯',
                title: 'Free 14-Day Trial',
                desc: 'Full feature access, no commitment',
              },
              {
                icon: '🤖',
                title: 'AI-Powered Insights',
                desc: 'Smart analysis and recommendations',
              },
              {
                icon: '🔒',
                title: 'Enterprise Security',
                desc: 'GDPR & HIPAA compliant',
              },
            ].map((feature) => (
              <div key={feature.title} className="flex items-start gap-4 group">
                <div className="text-2xl group-hover:scale-110 transition-transform flex-shrink-0">
                  {feature.icon}
                </div>
                <div>
                  <h3 className="text-slate-100 font-medium group-hover:text-white transition-colors">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-slate-400">{feature.desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-6 pt-8 border-t border-slate-700">
            {[
              { number: '10K+', label: 'Active Users' },
              { number: '50M+', label: 'Analyses' },
              { number: '99.9%', label: 'Uptime' },
            ].map((stat) => (
              <div key={stat.label}>
                <p className="text-2xl font-bold text-blue-400">{stat.number}</p>
                <p className="text-xs text-slate-400 mt-1">{stat.label}</p>
              </div>
            ))}
          </div>

          {/* Testimonials */}
          <div className="mt-12 pt-8 border-t border-slate-700 space-y-4">
            <p className="text-slate-400 text-sm">⭐ Trusted by data professionals</p>
            <div className="space-y-3">
              {[
                {
                  text: 'Best data analytics platform Ive used',
                  author: 'Sarah J.',
                },
                {
                  text: 'Saved our team hours every week',
                  author: 'Michael C.',
                },
              ].map((testimonial, idx) => (
                <blockquote key={idx} className="text-slate-300 italic text-sm">
                  "{testimonial.text}"
                  <footer className="text-slate-500 not-italic mt-1">
                    — {testimonial.author}
                  </footer>
                </blockquote>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ====================================================================
          Right Section - Registration Form
          ==================================================================== */}
      <div className="w-full lg:w-1/2 flex flex-col justify-center items-center p-6 sm:p-8 relative">
        {/* Background gradient */}
        <div className="absolute inset-0 opacity-5">
          <div className="absolute inset-0 bg-gradient-to-b from-blue-500 to-transparent" />
        </div>

        <div className="w-full max-w-md relative z-10">
          {/* Mobile Logo & Title */}
          <div className="lg:hidden mb-8 text-center">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center mx-auto mb-4">
              <BarChart3 className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-white">DataAnalytics</h1>
            <p className="text-slate-400 text-sm mt-2">
              Transform Data Into Insights
            </p>
          </div>

          {/* Form Header */}
          <div className="mb-8 hidden lg:block">
            <h2 className="text-3xl font-bold text-white mb-2">
              Create Your Account
            </h2>
            <p className="text-slate-400">
              Join thousands of teams making smarter decisions with data
            </p>
          </div>

          {/* Mobile Form Header */}
          <div className="mb-8 lg:hidden">
            <h2 className="text-2xl font-bold text-white mb-2">
              Get Started
            </h2>
            <p className="text-slate-400 text-sm">
              Create account and start exploring
            </p>
          </div>

          {/* Benefits Banner */}
          <div className="mb-6 p-4 bg-gradient-to-r from-blue-900/30 to-green-900/30 border border-blue-700/50 rounded-lg flex items-start gap-3">
            <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-slate-200">
                No credit card required
              </p>
              <p className="text-xs text-slate-400 mt-1">
                14-day free trial on any plan
              </p>
            </div>
          </div>

          {/* Registration Form */}
          <RegisterForm />

          {/* Divider */}
          <div className="relative my-8">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-slate-700" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-3 bg-slate-800 text-slate-400">
                Already have an account?
              </span>
            </div>
          </div>

          {/* Sign In Link */}
          <Link
            to="/login"
            className="w-full py-3 px-4 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500 transition-all duration-200 flex items-center justify-center gap-2 font-medium"
          >
            Sign In Instead
            <ArrowRight className="w-4 h-4" />
          </Link>

          {/* Footer Links */}
          <div className="mt-8 text-center space-y-4 pt-6 border-t border-slate-700">
            <div className="flex items-center justify-center gap-4 text-xs text-slate-500">
              <Link
                to="/privacy"
                className="hover:text-slate-400 transition-colors"
              >
                Privacy
              </Link>
              <span>•</span>
              <Link
                to="/terms"
                className="hover:text-slate-400 transition-colors"
              >
                Terms
              </Link>
              <span>•</span>
              <Link
                to="/contact"
                className="hover:text-slate-400 transition-colors"
              >
                Support
              </Link>
            </div>
          </div>

          {/* Security Badge */}
          <div className="mt-6 p-3 bg-slate-700/30 rounded-lg flex items-center justify-center gap-2 text-xs text-slate-400">
            <Shield className="w-4 h-4 text-green-400" />
            <span>SSL Encrypted • GDPR Compliant</span>
          </div>
        </div>
      </div>

      {/* ====================================================================
          Mobile Bottom Section
          ==================================================================== */}
      <div className="lg:hidden fixed bottom-0 left-0 right-0 bg-gradient-to-t from-slate-900 via-slate-900/80 to-transparent pt-8 pb-6 px-6 pointer-events-none">
        <p className="text-center text-xs text-slate-500">
          🔒 Your data is secure and encrypted
        </p>
      </div>
    </div>
  );
};

RegisterPage.displayName = 'RegisterPage';

export default RegisterPage;
