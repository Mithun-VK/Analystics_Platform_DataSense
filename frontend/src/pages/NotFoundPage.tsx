// src/pages/NotFoundPage.tsx

import { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Home,
  Search,
  ArrowRight,
  AlertCircle,
  HelpCircle,
  Mail,
  Phone,
  MapPin,
  ChevronRight,
  BarChart3,
  Database,
  Sparkles,
} from 'lucide-react';
import Button from '@/components/shared/Button';

interface Suggestion {
  icon: React.ElementType;
  title: string;
  description: string;
  action: string;
  path: string;
  color: string;
}

/**
 * NotFoundPage - 404 error page for invalid routes
 * Features: Animated 404 display, helpful suggestions, search, contact info
 * User-friendly error page with multiple recovery options
 */
const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchQuery, setSearchQuery] = useState('');
  const [isAnimating, setIsAnimating] = useState(false);

  // Trigger animation on mount
  useEffect(() => {
    setIsAnimating(true);
  }, []);

  // Helpful suggestions
  const suggestions: Suggestion[] = [
    {
      icon: Home,
      title: 'Go Home',
      description: 'Return to the main dashboard and start fresh',
      action: 'Dashboard',
      path: '/dashboard',
      color: 'from-blue-500 to-blue-600',
    },
    {
      icon: Database,
      title: 'Browse Datasets',
      description: 'Explore your datasets and recent uploads',
      action: 'View Datasets',
      path: '/datasets',
      color: 'from-purple-500 to-purple-600',
    },
    {
      icon: BarChart3,
      title: 'View Analytics',
      description: 'Check your analysis and insights',
      action: 'Analytics',
      path: '/eda',
      color: 'from-green-500 to-green-600',
    },
    {
      icon: Sparkles,
      title: 'Create Visualization',
      description: 'Start creating new data visualizations',
      action: 'Visualizations',
      path: '/visualizations',
      color: 'from-orange-500 to-orange-600',
    },
  ];

  // Handle search
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/search?q=${encodeURIComponent(searchQuery)}`);
    }
  };

  return (
    <div className="not-found-page">
      {/* Background decoration */}
      <div className="not-found-bg-decoration" />

      {/* Main content */}
      <div className="not-found-container">
        {/* 404 Display */}
        <div className={`not-found-display ${isAnimating ? 'animate' : ''}`}>
          <div className="not-found-number">
            <span className="not-found-digit">4</span>
            <div className="not-found-circle">
              <AlertCircle className="w-16 h-16" />
            </div>
            <span className="not-found-digit">4</span>
          </div>
        </div>

        {/* Message */}
        <div className="not-found-message">
          <h1 className="not-found-title">Page Not Found</h1>
          <p className="not-found-description">
            Sorry, the page you're looking for doesn't exist or has been moved.
          </p>
          <p className="not-found-path">
            Tried to access: <code>{location.pathname}</code>
          </p>
        </div>

        {/* Search */}
        <form onSubmit={handleSearch} className="not-found-search">
          <Search className="w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search or type a page name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="not-found-search-input"
          />
          <button
            type="submit"
            className="not-found-search-button"
            aria-label="Search"
          >
            <ArrowRight className="w-5 h-5" />
          </button>
        </form>

        {/* Quick Actions */}
        <div className="not-found-actions">
          <Button
            variant="primary"
            size="lg"
            leftIcon={Home}
            onClick={() => navigate('/dashboard')}
            className="not-found-home-button"
          >
            Go to Dashboard
          </Button>
          <Button
            variant="secondary"
            size="lg"
            onClick={() => navigate(-1)}
            className="not-found-back-button"
          >
            Go Back
          </Button>
        </div>

        {/* Suggestions Grid */}
        <div className="not-found-suggestions">
          <p className="not-found-suggestions-title">Here's where you can go:</p>
          <div className="not-found-suggestions-grid">
            {suggestions.map((suggestion) => {
              const Icon = suggestion.icon;
              return (
                <button
                  key={suggestion.path}
                  onClick={() => navigate(suggestion.path)}
                  className={`not-found-suggestion bg-gradient-to-br ${suggestion.color}`}
                >
                  <Icon className="w-6 h-6 text-white" />
                  <h3 className="not-found-suggestion-title">
                    {suggestion.title}
                  </h3>
                  <p className="not-found-suggestion-description">
                    {suggestion.description}
                  </p>
                  <span className="not-found-suggestion-action">
                    {suggestion.action}
                    <ChevronRight className="w-4 h-4 ml-2" />
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Help Section */}
        <div className="not-found-help">
          <div className="not-found-help-content">
            <div className="not-found-help-item">
              <HelpCircle className="w-5 h-5 text-blue-600" />
              <div>
                <p className="font-semibold text-gray-900">Need Help?</p>
                <p className="text-sm text-gray-600">
                  Check our documentation or contact support
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.open('/help', '_blank')}
            >
              Documentation
            </Button>
          </div>

          {/* Contact Info */}
          <div className="not-found-contact">
            <h3 className="not-found-contact-title">Can't find what you need?</h3>
            <div className="not-found-contact-methods">
              <a href="mailto:support@dataanalytics.com" className="not-found-contact-item">
                <Mail className="w-4 h-4" />
                <span>support@dataanalytics.com</span>
              </a>
              <a href="tel:+15551234567" className="not-found-contact-item">
                <Phone className="w-4 h-4" />
                <span>+1 (555) 123-4567</span>
              </a>
              <div className="not-found-contact-item">
                <MapPin className="w-4 h-4" />
                <span>San Francisco, CA 94105</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="not-found-footer">
          <p className="not-found-footer-text">
            Lost? Try using the search above or navigate using the menu
          </p>
          <div className="not-found-footer-links">
            <a href="/" className="not-found-footer-link">Home</a>
            <span className="not-found-footer-divider">•</span>
            <a href="/help" className="not-found-footer-link">Help</a>
            <span className="not-found-footer-divider">•</span>
            <a href="/contact" className="not-found-footer-link">Contact</a>
          </div>
        </div>
      </div>
    </div>
  );
};

NotFoundPage.displayName = 'NotFoundPage';

export default NotFoundPage;
