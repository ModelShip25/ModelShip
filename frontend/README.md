# ModelShip Frontend

Modern React TypeScript frontend for the ModelShip AI-powered auto data labeling platform.

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ 
- npm or yarn
- ModelShip backend running on `http://localhost:8000`

### Installation
```bash
# Install dependencies
npm install

# Start development server
npm start
```

The frontend will be available at `http://localhost:3000`

## 🏗️ Architecture

### Tech Stack
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling and design system
- **React Router** - Client-side routing
- **React Query** - Server state management
- **Axios** - HTTP client
- **Heroicons** - Icon library
- **Headless UI** - Accessible UI components
- **React Hot Toast** - Notifications

### Project Structure
```
src/
├── components/          # Reusable UI components
│   ├── Layout/         # Layout components (Header, Sidebar)
│   └── ...
├── contexts/           # React contexts (Auth, etc.)
├── pages/              # Page components
├── services/           # API services and utilities
├── types/              # TypeScript type definitions
└── utils/              # Helper functions
```

## 🎨 Features

### ✅ Implemented
- **Authentication System** - Login/logout with JWT tokens
- **Modern Dashboard** - Overview of projects, stats, and quick actions
- **Responsive Layout** - Mobile-friendly sidebar and header
- **Navigation** - Complete routing structure
- **API Integration** - Connected to ModelShip backend (98 endpoints)
- **Loading States** - Proper loading and error handling
- **Toast Notifications** - User feedback system

### 🚧 Coming Soon
- **Project Management** - Create, edit, and manage labeling projects
- **File Upload Interface** - Drag-and-drop file uploads
- **Classification Interface** - Real-time image and text classification
- **Review System** - Human-in-the-loop label review
- **Analytics Dashboard** - Data insights and performance metrics
- **Export Interface** - Download labeled data in multiple formats
- **Team Management** - User roles and collaboration features

## 🔌 API Integration

The frontend is fully integrated with the ModelShip backend API:

- **Authentication** - Login, register, user management
- **Projects** - CRUD operations for labeling projects
- **Files** - Upload and manage datasets
- **Classification** - Image and text classification
- **Review** - Human review workflows
- **Analytics** - Performance metrics and insights
- **Export** - Data export in multiple formats

## 🎯 Key Components

### Layout System
- **Sidebar Navigation** - Collapsible sidebar with project navigation
- **Header** - Search, notifications, and user menu
- **Protected Routes** - Authentication-required pages

### Dashboard
- **Stats Cards** - Key metrics overview
- **Quick Actions** - Fast access to common tasks
- **Recent Projects** - Latest project activity
- **Welcome Section** - Personalized user greeting

### Authentication
- **Login Page** - Secure authentication with JWT
- **Auth Context** - Global authentication state
- **Protected Routes** - Automatic redirection for unauthenticated users

## 🌈 Design System

### Colors
- **Primary**: Blue (#3b82f6) - ModelShip brand color
- **Gray Scale**: Modern neutral palette
- **Status Colors**: Success, warning, error states

### Typography
- **Font**: System fonts for optimal performance
- **Hierarchy**: Clear heading and text styles
- **Responsive**: Scales across device sizes

### Components
- **Cards**: Consistent shadow and spacing
- **Buttons**: Hover effects and loading states
- **Forms**: Accessible form controls
- **Icons**: Heroicons for consistency

## 🔧 Development

### Available Scripts
- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run test suite
- `npm run eject` - Eject from Create React App

### Environment Variables
Create a `.env.local` file:
```
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
REACT_APP_UPLOAD_MAX_SIZE=10485760
```

### Code Style
- **TypeScript** - Strict type checking enabled
- **ESLint** - Code linting and formatting
- **Prettier** - Code formatting (recommended)

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Deploy Options
- **Vercel** - Recommended for React apps
- **Netlify** - Static site hosting
- **AWS S3 + CloudFront** - Enterprise deployment
- **Docker** - Containerized deployment

## 🤝 Contributing

1. Follow the existing code structure
2. Use TypeScript for all new components
3. Implement proper error handling
4. Add loading states for async operations
5. Follow the established design system

## 📚 Resources

- [React Documentation](https://reactjs.org/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [ModelShip API Guide](../backend/API_INTEGRATION_GUIDE.md)
