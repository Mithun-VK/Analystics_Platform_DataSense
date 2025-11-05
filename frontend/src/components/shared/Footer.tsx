// src/components/shared/Footer.tsx

/**
 * Footer Component - Modern, Clean Design
 * ✅ ENHANCED: Responsive layout, dark mode, smooth animations, accessibility
 * Features: Newsletter signup, social links, multiple sections, company info
 */

import { useState, useCallback, memo } from 'react';
import { Link } from 'react-router-dom';
import {
  Mail,
  Phone,
  MapPin,
  Github,
  Twitter,
  Linkedin,
  Facebook,
  ArrowRight,
  Heart,
  Send,
} from 'lucide-react';

// ============================================================================
// Type Definitions
// ============================================================================

interface FooterLink {
  label: string;
  path: string;
}

interface FooterSection {
  title: string;
  links: FooterLink[];
}

interface FooterProps {
  showNewsletter?: boolean;
  showSocial?: boolean;
  sections?: FooterSection[];
  companyName?: string;
  foundedYear?: number;
  contactInfo?: {
    email?: string;
    phone?: string;
    address?: string;
  };
}

// ============================================================================
// Memoized Sub-Components
// ============================================================================

/**
 * ✅ Newsletter Section Component
 */
interface NewsletterSectionProps {
  onSubmit: (email: string) => void;
}

const NewsletterSection = memo<NewsletterSectionProps>(({ onSubmit }) => {
  const [email, setEmail] = useState('');
  const [subscribed, setSubscribed] = useState(false);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (email) {
        onSubmit(email);
        setSubscribed(true);
        setEmail('');
        setTimeout(() => setSubscribed(false), 3000);
      }
    },
    [email, onSubmit]
  );

  return (
    <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 sm:p-10 shadow-lg">
      <div className="max-w-md mx-auto sm:mx-0">
        <h3 className="text-2xl font-bold text-white mb-2">Stay Updated</h3>
        <p className="text-blue-100 mb-6">
          Get the latest updates, tips, and insights delivered to your inbox.
        </p>

        <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-2">
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="flex-1 px-4 py-3 bg-white/20 backdrop-blur-sm border border-white/30 rounded-lg text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-white focus:bg-white/30 transition-all"
            aria-label="Email address"
            required
          />
          <button
            type="submit"
            disabled={subscribed}
            className={`px-6 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-all whitespace-nowrap ${
              subscribed
                ? 'bg-white/20 text-white'
                : 'bg-white text-blue-600 hover:bg-blue-50 shadow-lg hover:shadow-xl'
            }`}
            aria-label="Subscribe to newsletter"
          >
            {subscribed ? (
              <>
                <span className="text-sm">Subscribed!</span>
              </>
            ) : (
              <>
                <span className="hidden sm:inline">Subscribe</span>
                <Send className="w-4 h-4" />
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
});

NewsletterSection.displayName = 'NewsletterSection';

/**
 * ✅ Footer Logo Component
 */
interface FooterLogoProps {
  companyName: string;
}

const FooterLogo = memo<FooterLogoProps>(({ companyName }) => (
  <Link
    to="/"
    className="inline-flex items-center gap-3 group hover:opacity-80 transition-opacity"
  >
    <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-md group-hover:shadow-lg transition-shadow">
      <svg
        className="w-7 h-7 text-white"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
        />
      </svg>
    </div>
    <span className="text-lg font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
      {companyName}
    </span>
  </Link>
));

FooterLogo.displayName = 'FooterLogo';

/**
 * ✅ Contact Info Component
 */
interface ContactInfoProps {
  email?: string;
  phone?: string;
  address?: string;
}

const ContactInfo = memo<ContactInfoProps>(({ email, phone, address }) => (
  <div className="space-y-4">
    {email && (
      <a
        href={`mailto:${email}`}
        className="flex items-center gap-3 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors group"
      >
        <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center group-hover:bg-blue-200 dark:group-hover:bg-blue-800/50 transition-colors">
          <Mail className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        </div>
        <span className="text-sm font-medium">{email}</span>
      </a>
    )}

    {phone && (
      <a
        href={`tel:${phone}`}
        className="flex items-center gap-3 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors group"
      >
        <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center group-hover:bg-blue-200 dark:group-hover:bg-blue-800/50 transition-colors">
          <Phone className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        </div>
        <span className="text-sm font-medium">{phone}</span>
      </a>
    )}

    {address && (
      <div className="flex items-start gap-3 text-gray-600 dark:text-gray-400">
        <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
          <MapPin className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        </div>
        <span className="text-sm font-medium pt-0.5">{address}</span>
      </div>
    )}
  </div>
));

ContactInfo.displayName = 'ContactInfo';

/**
 * ✅ Footer Links Column Component
 */
interface FooterColumnProps {
  title: string;
  links: FooterLink[];
}

const FooterColumn = memo<FooterColumnProps>(({ title, links }) => (
  <div>
    <h4 className="font-bold text-gray-900 dark:text-white mb-4 text-sm uppercase tracking-wider">
      {title}
    </h4>
    <ul className="space-y-3">
      {links.map((link) => (
        <li key={link.path}>
          <Link
            to={link.path}
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors font-medium flex items-center gap-1 group"
          >
            <ArrowRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transform -translate-x-2 group-hover:translate-x-0 transition-all" />
            <span>{link.label}</span>
          </Link>
        </li>
      ))}
    </ul>
  </div>
));

FooterColumn.displayName = 'FooterColumn';

/**
 * ✅ Social Links Component
 */
interface SocialLink {
  icon: React.ElementType;
  href: string;
  label: string;
}

interface SocialLinksProps {
  links: SocialLink[];
}

const SocialLinks = memo<SocialLinksProps>(({ links }) => (
  <div>
    <h4 className="font-bold text-gray-900 dark:text-white mb-4 text-sm uppercase tracking-wider">
      Follow Us
    </h4>
    <div className="flex items-center gap-3">
      {links.map((link) => {
        const Icon = link.icon;
        return (
          <a
            key={link.label}
            href={link.href}
            target="_blank"
            rel="noopener noreferrer"
            className="w-10 h-10 bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center text-gray-700 dark:text-gray-300 hover:bg-blue-600 hover:text-white dark:hover:bg-blue-600 dark:hover:text-white transition-colors"
            aria-label={link.label}
          >
            <Icon className="w-5 h-5" />
          </a>
        );
      })}
    </div>
  </div>
));

SocialLinks.displayName = 'SocialLinks';

/**
 * ✅ Footer Bottom Bar Component
 */
interface FooterBottomProps {
  companyName: string;
  foundedYear: number;
  currentYear: number;
}

const FooterBottom = memo<FooterBottomProps>(
  ({ companyName, foundedYear, currentYear }) => (
    <div className="border-t border-gray-200 dark:border-gray-700 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        {/* Copyright */}
        <p className="text-sm text-gray-600 dark:text-gray-400 flex flex-col sm:flex-row items-start sm:items-center gap-2">
          <span>
            © {foundedYear === currentYear ? currentYear : `${foundedYear}-${currentYear}`}
          </span>
          <span className="hidden sm:inline text-gray-400">•</span>
          <span className="font-semibold text-gray-900 dark:text-white">{companyName}</span>
          <span className="hidden sm:inline text-gray-400">•</span>
          <span className="flex items-center gap-1">
            Made with
            <Heart className="w-4 h-4 text-red-500 fill-red-500" />
            by the team
          </span>
        </p>

        {/* Bottom Links */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 md:justify-end">
          <Link
            to="/privacy"
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
          >
            Privacy Policy
          </Link>
          <span className="hidden sm:inline text-gray-300">•</span>
          <Link
            to="/terms"
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
          >
            Terms of Service
          </Link>
          <span className="hidden sm:inline text-gray-300">•</span>
          <Link
            to="/cookies"
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
          >
            Cookie Policy
          </Link>
        </div>
      </div>
    </div>
  )
);

FooterBottom.displayName = 'FooterBottom';

// ============================================================================
// Main Footer Component
// ============================================================================

/**
 * ✅ Main Footer Component - Enhanced & Clean
 */
const Footer = memo<FooterProps>(
  ({
    showNewsletter = true,
    showSocial = true,
    sections,
    companyName = 'DataSense',
    foundedYear = 2025,
    contactInfo = {
      email: 'support@datasense.app',
      phone: '+1 (555) 123-4567',
      address: 'San Francisco, CA 94105',
    },
  }) => {
    const currentYear = new Date().getFullYear();

    // Default footer sections
    const defaultSections: FooterSection[] = [
      {
        title: 'Product',
        links: [
          { label: 'Features', path: '/features' },
          { label: 'Pricing', path: '/pricing' },
          { label: 'Security', path: '/security' },
          { label: 'Roadmap', path: '/roadmap' },
        ],
      },
      {
        title: 'Company',
        links: [
          { label: 'About', path: '/about' },
          { label: 'Blog', path: '/blog' },
          { label: 'Careers', path: '/careers' },
          { label: 'Contact', path: '/contact' },
        ],
      },
      {
        title: 'Resources',
        links: [
          { label: 'Documentation', path: '/docs' },
          { label: 'API Reference', path: '/api' },
          { label: 'Tutorials', path: '/tutorials' },
          { label: 'Support', path: '/support' },
        ],
      },
    ];

    const footerSections = sections || defaultSections;

    const socialLinks: SocialLink[] = [
      { icon: Github, href: 'https://github.com', label: 'GitHub' },
      { icon: Twitter, href: 'https://twitter.com', label: 'Twitter' },
      { icon: Linkedin, href: 'https://linkedin.com', label: 'LinkedIn' },
      { icon: Facebook, href: 'https://facebook.com', label: 'Facebook' },
    ];

    const handleNewsletterSubmit = useCallback((email: string) => {
      console.log('Newsletter subscription:', email);
      // Add your API call here
    }, []);

    return (
      <footer
        className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700"
        role="contentinfo"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
          {/* Newsletter Section */}
          {showNewsletter && (
            <div className="mb-16">
              <NewsletterSection onSubmit={handleNewsletterSubmit} />
            </div>
          )}

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-12 mb-12">
            {/* Brand & Contact - 3 columns */}
            <div className="md:col-span-3 space-y-6">
              <FooterLogo companyName={companyName} />
              <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                Transform your data into actionable insights with our comprehensive
                analytics platform. Explore, analyze, and visualize like never before.
              </p>
              <ContactInfo
                email={contactInfo?.email}
                phone={contactInfo?.phone}
                address={contactInfo?.address}
              />
            </div>

            {/* Footer Sections - 9 columns */}
            <div className="md:col-span-9">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                {footerSections.map((section) => (
                  <FooterColumn
                    key={section.title}
                    title={section.title}
                    links={section.links}
                  />
                ))}

                {/* Social Links */}
                {showSocial && <SocialLinks links={socialLinks} />}
              </div>
            </div>
          </div>

          {/* Footer Bottom */}
          <FooterBottom
            companyName={companyName}
            foundedYear={foundedYear}
            currentYear={currentYear}
          />
        </div>
      </footer>
    );
  }
);

Footer.displayName = 'Footer';

export default Footer;
