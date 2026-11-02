# web/Dockerfile
FROM node:22-alpine

WORKDIR /app

# Install dependencies first for layer caching
COPY package*.json ./
RUN npm ci

# Copy source code and build production bundle
COPY . .
RUN npm run build

# Install a lightweight static server
RUN npm install -g serve

EXPOSE 3000

# -s flags single-page application (SPA fallback to index.html for React Router)
# -l sets the port from the environment variable (defaults to 3000)
CMD ["sh", "-c", "serve -s dist -l ${WEB_INTERNAL_PORT:-3000}"]