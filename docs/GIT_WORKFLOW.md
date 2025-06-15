# Git Workflow and Branching Strategy

## Branch Types

- `main` - Production-ready code
- `develop` - Integration branch for feature development
- `feature/*` - New features and non-emergency bug fixes
- `hotfix/*` - Emergency production fixes
- `release/*` - Release preparation

## Branch Naming Convention

- Feature branches: `feature/descriptive-name`
- Hotfix branches: `hotfix/issue-description`
- Release branches: `release/vX.Y.Z`

## Workflow

1. **Main Branch (`main`)**
   - Contains production-ready code
   - Protected branch - requires PR and approvals
   - Tagged with version numbers for releases

2. **Development Branch (`develop`)**
   - Integration branch for features
   - Protected branch - requires PR
   - Base branch for feature development

3. **Feature Branches (`feature/*`)**
   - Created from: `develop`
   - Merge back into: `develop`
   - Naming: `feature/descriptive-name`
   - Example: `feature/add-trading-dashboard`

4. **Hotfix Branches (`hotfix/*`)**
   - Created from: `main`
   - Merge back into: `main` AND `develop`
   - Naming: `hotfix/issue-description`
   - Example: `hotfix/fix-api-authentication`

5. **Release Branches (`release/*`)**
   - Created from: `develop`
   - Merge back into: `main` AND `develop`
   - Naming: `release/vX.Y.Z`
   - Example: `release/v1.2.0`

## Commit Message Convention

Format: `type(scope): description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

Example:
```
feat(trading): add real-time price alerts
fix(auth): resolve JWT token expiration issue
docs(api): update trading endpoints documentation
```

## Pull Request Process

1. Create PR from your feature branch to `develop`
2. Fill out PR template
3. Request reviews from team members
4. Address review comments
5. Squash and merge when approved

## Release Process

1. Create release branch from `develop`
2. Version bump and changelog update
3. Bug fixes and documentation updates
4. Create PR to `main`
5. After merge, tag release in `main`
6. Merge back into `develop`

## Protected Branch Rules

### Main Branch (`main`)
- Require pull request reviews
- Require status checks to pass
- No direct pushes
- Linear history (no merge commits)

### Develop Branch (`develop`)
- Require pull request reviews
- Require status checks to pass
- No direct pushes

## Git Commands Quick Reference

```bash
# Start a new feature
git checkout develop
git pull origin develop
git checkout -b feature/my-feature

# Update feature branch with develop
git checkout feature/my-feature
git fetch origin
git rebase origin/develop

# Create a hotfix
git checkout main
git pull origin main
git checkout -b hotfix/critical-fix

# Create a release
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0
```

## CI/CD Integration

The branching strategy is integrated with our CI/CD pipeline:
- Feature branches: Run tests and checks
- Develop: Deploy to staging
- Main: Deploy to production
- Release branches: Run additional tests and checks

## Best Practices

1. Keep branches short-lived
2. Regularly update feature branches with `develop`
3. Write descriptive commit messages
4. Delete branches after merging
5. Use rebase instead of merge when updating feature branches
6. Create atomic commits (one logical change per commit)
7. Never force push to protected branches 