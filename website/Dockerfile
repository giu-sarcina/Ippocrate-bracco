# use the official Bun image
# see all versions at https://hub.docker.com/r/oven/bun/tags
FROM oven/bun:1 AS base
WORKDIR /usr/app

# install dependencies into temp directory
# this will cache them and speed up future builds
FROM base AS install
COPY src /usr/app/src
COPY package.json bun.lock /usr/app
COPY build.ts .
RUN ls
RUN bun install --frozen-lockfile --production

# Declare the build argument
ARG EMAIL="placeholder@gmail.com"
ARG NAME="Ippocrate"

# overwrite environment variables for production build
ENV NODE_ENV=production
ENV EMAIL=${EMAIL}
ENV NAME=${NAME}
# create a constants file with the build arguments
RUN echo "export const EMAIL = \"${EMAIL}\";" > src/constants.tsx && \
    echo "export const NAME = \"${NAME}\";" >> src/constants.tsx && \
    cat src/constants.tsx 

# build
RUN bun run build src/index.tsx --outdir=dist --minify --source-map=linked --target=browser

# copy production dependencies and source code into final image
FROM base AS release
COPY --from=install /usr/app/node_modules node_modules
COPY --from=install /usr/app/dist dist

# run the app
USER bun
EXPOSE 3000/tcp
ENTRYPOINT [ "bunx", "serve", "dist", "-l", "tcp://0.0.0.0:3000" ]